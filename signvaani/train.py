"""Train a sign classifier.

Examples:
    # Signer-independent: test on signer s05, validate on s04
    python -m signvaani.train --data data/keypoints --test-signers s05 --val-signers s04

    # Leave-one-signer-out cross-validation (the number to report)
    python -m signvaani.train --data data/keypoints --loso

    # Signer-dependent random split (report only next to the LOSO result)
    python -m signvaani.train --data data/keypoints --random-split
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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
    test_acc = None
    if test:
        _, test_acc = run_epoch(model, mk(test, False), device, criterion)
        print(f"TEST accuracy: {test_acc:.4f}  ({len(test)} clips)")
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": args.model, "state_dict": best_state, "labels": labels,
                    "seq_len": args.seq_len, "feature_dim": FEATURE_DIM}, out_path)
        print(f"saved {out_path}")
    return best_acc, test_acc


def main():
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
    args = p.parse_args()

    clips = index_clips(args.data)
    labels = sorted({c.sign for c in clips})
    people = signers(clips)
    print(f"{len(clips)} clips, {len(labels)} classes, {len(people)} signers: {people}")

    if args.loso:
        results = {}
        for s in people:
            print(f"\n===== fold: test signer {s} =====")
            train, val, test = split_by_signer(clips, [s], seed=args.seed)
            results[s] = train_one(train, val, test, labels, args)[1]
        accs = np.array(list(results.values()))
        print("\nLOSO results:", {k: round(v, 4) for k, v in results.items()})
        print(f"LOSO mean top-1 = {accs.mean():.4f} +/- {accs.std():.4f}")
        Path("runs").mkdir(exist_ok=True)
        out = Path("runs") / f"loso_{args.model}_{int(time.time())}.json"
        out.write_text(json.dumps({"per_signer": results, "mean": accs.mean(), "std": accs.std(),
                                   "args": vars(args)}, indent=2))
        print(f"saved {out}")
        return

    if args.random_split:
        train, val, test = random_split(clips, seed=args.seed)
    else:
        if not args.test_signers:
            p.error("give --test-signers (signer-independent), --loso, or --random-split")
        train, val, test = split_by_signer(clips, args.test_signers, args.val_signers, seed=args.seed)
    print(f"train {len(train)} / val {len(val)} / test {len(test)} clips")
    train_one(train, val, test, labels, args, Path(args.out))


if __name__ == "__main__":
    main()
