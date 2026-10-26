"""Train a sign classifier.

Examples:
    # Signer-independent: test on signer s05, validate on s04
    python -m signvaani.train --data data/keypoints --test-signers s05 --val-signers s04

    # Leave-one-signer-out cross-validation (the number to report)
    python -m signvaani.train --data data/keypoints --loso

    # Signer-dependent random split (report only next to the LOSO result)
    python -m signvaani.train --data data/keypoints --random-split

    # INCLUDE-50, official split (real public ISL data; the split name stands in for the signer)
    python -m signvaani.train --data data/include_keypoints/include50 --test-signers test --val-signers val

Every run writes runs/<mode>_<model>_<time>.json: top-1 and macro-F1 per fold and the prediction for
every test clip. `python -m signvaani.confusion <run.json>` turns it into confusion matrices and the
most confused sign pairs (Day 3).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from .data import SignDataset, index_clips, random_split, signers, split_by_signer
from .landmarks import FEATURE_DIM
from .models import build_model


def run_epoch(model, loader, device, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
    return loss_sum / max(total, 1), correct / max(total, 1)


def train_one(train, val, test, labels, args, out_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    mk = lambda clips, tr: DataLoader(SignDataset(clips, labels, args.seq_len, train=tr, seed=args.seed),
                                      batch_size=args.batch_size, shuffle=tr, num_workers=args.workers)
    train_dl, val_dl = mk(train, True), mk(val, False)
    model = build_model(args.model, FEATURE_DIM, len(labels)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_acc, best_state, stale = -1.0, None, 0
    for epoch in range(1, args.epochs + 1):
        tl, ta = run_epoch(model, train_dl, device, criterion, opt)
        vl, va = run_epoch(model, val_dl, device, criterion)
        sched.step()
        print(f"epoch {epoch:3d}  train loss {tl:.3f} acc {ta:.3f}  |  val loss {vl:.3f} acc {va:.3f}")
        if va > best_acc:
            best_acc, stale = va, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    result = None
    if test:
        result = predict(model, mk(test, False), test, labels, device)
        print(f"TEST accuracy: {result['top1']:.4f}  macro-F1 {result['macro_f1']:.4f}  ({len(test)} clips)")
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": args.model, "state_dict": best_state, "labels": labels,
                    "seq_len": args.seq_len, "feature_dim": FEATURE_DIM}, out_path)
        print(f"saved {out_path}")
    return best_acc, result


def predict(model, loader, clips, labels, device):
    """Top-1, macro-F1 and one record per clip (in loader order, which must not shuffle)."""
    model.eval()
    probs = []
    with torch.no_grad():
        for x, _ in loader:
            probs.append(torch.softmax(model(x.to(device)), 1).cpu().numpy())
    probs = np.concatenate(probs)
    y = np.array([labels.index(c.sign) for c in clips])
    pred = probs.argmax(1)
    records = [{"clip": c.path.as_posix(), "signer": c.signer, "true": c.sign, "pred": labels[k],
                "conf": round(float(p[k]), 4), "top3": [labels[j] for j in np.argsort(-p)[:3]]}
               for c, p, k in zip(clips, probs, pred)]
    return {"top1": float((pred == y).mean()), "n": len(clips), "predictions": records,
            "macro_f1": float(f1_score(y, pred, labels=np.unique(y), average="macro", zero_division=0))}


def save_run(mode, folds, labels, args):
    """runs/<mode>_<model>_<time>.json with the summary and every fold's predictions."""
    top1 = np.array([f["top1"] for f in folds])
    run = {"mode": mode, "model": args.model, "data": Path(args.data).as_posix(), "labels": labels,
           "parameters": sum(p.numel() for p in build_model(args.model, FEATURE_DIM, len(labels)).parameters()),
           "mean": float(top1.mean()), "std": float(top1.std()),
           "macro_f1_mean": float(np.mean([f["macro_f1"] for f in folds])),
           "per_signer": {f["name"]: f["top1"] for f in folds}, "folds": folds, "args": vars(args)}
    Path(args.runs).mkdir(parents=True, exist_ok=True)
    out = Path(args.runs) / f"{mode}_{args.model}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(run, indent=1))
    print(f"saved {out}  (next: python -m signvaani.confusion {out.as_posix()})")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/keypoints")
    p.add_argument("--model", default="gru", choices=["gru", "transformer", "cnn"])
    p.add_argument("--test-signers", nargs="*", default=[])
    p.add_argument("--val-signers", nargs="*", default=[])
    p.add_argument("--loso", action="store_true", help="leave-one-signer-out cross-validation")
    p.add_argument("--random-split", action="store_true", help="signer-dependent random split")
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="models/best.pt")
    p.add_argument("--runs", default="runs", help="folder for the run record (.json)")
    args = p.parse_args(argv)

    clips = index_clips(args.data)
    labels = sorted({c.sign for c in clips})
    people = signers(clips)
    print(f"{len(clips)} clips, {len(labels)} classes, {len(people)} signers: {people}")

    if args.loso:
        if set(people) <= {"train", "val", "test"}:
            p.error("these folders hold an official split (INCLUDE), not signers: use "
                    "--test-signers test --val-signers val instead of --loso")
        folds = []
        for s in people:
            print(f"\n===== fold: test signer {s} =====")
            train, val, test = split_by_signer(clips, [s], seed=args.seed)
            val_acc, result = train_one(train, val, test, labels, args)
            folds.append({"name": s, "test_signers": [s], "val_acc": val_acc, **result})
        print("\nLOSO top-1 per test signer:")
        for f in folds:
            print(f"  {f['name']}: {f['top1']:.4f}  (macro-F1 {f['macro_f1']:.4f}, {f['n']} clips)")
        top1 = np.array([f["top1"] for f in folds])
        print(f"LOSO mean top-1 = {top1.mean():.4f} +/- {top1.std():.4f}, "
              f"macro-F1 {np.mean([f['macro_f1'] for f in folds]):.4f}")
        if top1.mean() < 0.6:
            print("LOSO mean is below 60%: apply fallback F2 / F3 before Day 4 (docs/day3.md)")
        save_run("loso", folds, labels, args)
        return

    if args.random_split:
        mode, name = "random", "random"
        train, val, test = random_split(clips, seed=args.seed)
    else:
        if not args.test_signers:
            p.error("give --test-signers (signer-independent), --loso, or --random-split")
        mode, name = "signer", "+".join(args.test_signers)
        train, val, test = split_by_signer(clips, args.test_signers, args.val_signers, seed=args.seed)
    print(f"train {len(train)} / val {len(val)} / test {len(test)} clips")
    val_acc, result = train_one(train, val, test, labels, args, Path(args.out))
    if result:
        save_run(mode, [{"name": name, "test_signers": sorted({c.signer for c in test}), "val_acc": val_acc,
                         **result}], labels, args)


if __name__ == "__main__":
    main()
