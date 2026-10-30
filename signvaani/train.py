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

    # Final model for the app (Day 4): every signer trains except a validation signer, no test set
    python -m signvaani.train --data data/keypoints --final --val-signers s05 --out models/best.pt

    # Ablation (Day 4, Table R4): landmark groups, augmentation and velocity features can be switched off
    python -m signvaani.train --data data/keypoints --loso --parts hands pose --no-augment --no-velocity

Two defaults make the model behave better live (switch off with --no-balance / --no-trim):
  - signs with fewer training clips get a larger weight in the loss, so a sign with 6 clips counts as
    much as one with 15 and the model does not fall back to the biggest classes;
  - every clip is cut to the window the live spotter would classify (from just before the hands move to
    soon after they stop), so training clips look like the ones the app sees.

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
from .landmarks import ALL_GROUPS, FEATURE_DIM
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


def class_weights(clips, labels):
    """Loss weight per label, inversely proportional to its number of clips (1 when all are equal)."""
    counts = np.array([sum(c.sign == label for c in clips) for label in labels], np.float64)
    weights = np.where(counts > 0, counts.sum() / (len(labels) * np.maximum(counts, 1)), 0.0)
    return torch.tensor(weights, dtype=torch.float32)


def train_one(train, val, test, labels, args, out_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    mk = lambda clips, tr: DataLoader(SignDataset(clips, labels, args.seq_len, train=tr, seed=args.seed,
                                                  trim=args.trim, augment=args.augment, parts=args.parts,
                                                  velocity=args.velocity),
                                      batch_size=args.batch_size, shuffle=tr, num_workers=args.workers)
    train_dl, val_dl = mk(train, True), mk(val, False)
    model = build_model(args.model, FEATURE_DIM, len(labels)).to(device)
    weights = class_weights(train, labels) if args.balance else None
    if weights is not None:
        print("class weights: " + ", ".join(f"{l} {w:.2f}" for l, w in zip(labels, weights.tolist())))
    criterion = nn.CrossEntropyLoss(weight=None if weights is None else weights.to(device), label_smoothing=0.1)
    eval_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_acc, best_state, stale = -1.0, None, 0
    for epoch in range(1, args.epochs + 1):
        tl, ta = run_epoch(model, train_dl, device, criterion, opt)
        vl, va = run_epoch(model, val_dl, device, eval_criterion)
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
                    "seq_len": args.seq_len, "feature_dim": FEATURE_DIM, "trim": args.trim,
                    "parts": list(args.parts), "velocity": args.velocity}, out_path)
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


def save_run(mode, folds, labels, args, tag=""):
    """runs/<mode>_<model><tag>_<time>.json with the summary and every fold's predictions."""
    top1 = np.array([f["top1"] for f in folds])
    run = {"mode": mode, "model": args.model, "data": Path(args.data).as_posix(), "labels": labels,
           "parameters": sum(p.numel() for p in build_model(args.model, FEATURE_DIM, len(labels)).parameters()),
           "mean": float(top1.mean()), "std": float(top1.std()),
           "macro_f1_mean": float(np.mean([f["macro_f1"] for f in folds])),
           "per_signer": {f["name"]: f["top1"] for f in folds}, "folds": folds,
           "args": {k: list(v) if isinstance(v, tuple) else v for k, v in vars(args).items()}}
    Path(args.runs).mkdir(parents=True, exist_ok=True)
    stem = f"{mode}_{args.model}{tag}_{time.strftime('%Y%m%d-%H%M%S')}"
    out, n = Path(args.runs) / f"{stem}.json", 1
    while out.exists():  # two runs in the same second (e.g. signvaani.experiments) keep both records
        out, n = Path(args.runs) / f"{stem}-{n}.json", n + 1
    out.write_text(json.dumps(run, indent=1))
    print(f"saved {out}  (next: python -m signvaani.confusion {out.as_posix()})")
    return out


OFFICIAL_SPLIT = {"train", "val", "test"}  # INCLUDE folders: the split name stands in for the signer


def loso(clips, labels, args):
    """Leave-one-signer-out: one fold per signer, each tested on the signer it left out."""
    folds = []
    for s in signers(clips):
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
    return folds


def one_split(clips, labels, args, test_signers=(), val_signers=(), random=False, out_path=None):
    """A single train / val / test split -> [fold], or [] when there is no test set."""
    if random:
        name, (train, val, test) = "random", random_split(clips, seed=args.seed)
    else:
        name = "+".join(test_signers)
        train, val, test = split_by_signer(clips, test_signers, val_signers, seed=args.seed)
    print(f"train {len(train)} / val {len(val)} / test {len(test)} clips")
    val_acc, result = train_one(train, val, test, labels, args, out_path)
    if not result:
        return []
    return [{"name": name, "test_signers": sorted({c.signer for c in test}), "val_acc": val_acc, **result}]


def make_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/keypoints")
    p.add_argument("--model", default="gru", choices=["gru", "transformer", "cnn"])
    p.add_argument("--test-signers", nargs="*", default=[])
    p.add_argument("--val-signers", nargs="*", default=[])
    p.add_argument("--loso", action="store_true", help="leave-one-signer-out cross-validation")
    p.add_argument("--random-split", action="store_true", help="signer-dependent random split")
    p.add_argument("--final", action="store_true",
                   help="train the model for the app on every signer except --val-signers (no test set)")
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="models/best.pt")
    p.add_argument("--runs", default="runs", help="folder for the run record (.json)")
    p.add_argument("--no-balance", dest="balance", action="store_false",
                   help="do not weight the loss by class size")
    p.add_argument("--no-trim", dest="trim", action="store_false",
                   help="train on whole clips instead of the window the live spotter cuts out")
    p.add_argument("--parts", nargs="+", choices=ALL_GROUPS, default=list(ALL_GROUPS),
                   help="landmark groups the model sees (ablation): hands, pose, face")
    p.add_argument("--no-augment", dest="augment", action="store_false", help="ablation: no augmentation")
    p.add_argument("--no-velocity", dest="velocity", action="store_false",
                   help="ablation: positions only, velocity features set to zero")
    return p


def main(argv=None):
    p = make_parser()
    args = p.parse_args(argv)

    clips = index_clips(args.data)
    labels = sorted({c.sign for c in clips})
    people = signers(clips)
    print(f"{len(clips)} clips, {len(labels)} classes, {len(people)} signers: {people}")

    if args.final:
        if args.loso or args.random_split or args.test_signers:
            p.error("--final trains the model for the app and has no test set; report LOSO numbers instead")
        print("final model: " + (f"validation signers {args.val_signers}" if args.val_signers
                                 else "15% of the clips for validation"))
        one_split(clips, labels, args, val_signers=args.val_signers, out_path=Path(args.out))
        return

    if args.loso:
        if set(people) <= OFFICIAL_SPLIT:
            p.error("these folders hold an official split (INCLUDE), not signers: use "
                    "--test-signers test --val-signers val instead of --loso")
        folds = loso(clips, labels, args)
        if np.mean([f["top1"] for f in folds]) < 0.6:
            print("LOSO mean is below 60%: apply fallback F2 / F3 before Day 4 (docs/day3.md)")
        save_run("loso", folds, labels, args)
        return

    if not args.random_split and not args.test_signers:
        p.error("give --test-signers (signer-independent), --loso, --random-split or --final")
    folds = one_split(clips, labels, args, args.test_signers, args.val_signers, args.random_split, Path(args.out))
    if folds:
        save_run("random" if args.random_split else "signer", folds, labels, args)


if __name__ == "__main__":
    main()
