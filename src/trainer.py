import enum
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import VideoPathsDataLoader
from loss import HardNegativeNCE
from metrics import RecallAtK
import datetime
import os
import time



class Phase(enum.Enum):
    TRAINING   = 1
    VALIDATION = 2

def one_epoch(phase, model, loader, optimizer=None):
    """
    Single pass over loader.
    If phase==TRAINING, do backward/update; otherwise just eval and
    accumulate loss.
    """
    training = (phase is Phase.TRAINING)
    model.module.head.train(training) # set head only
    total_loss = 0.0
    total_samples = 0

    device = model.module.device

    # instantiate HN‐NCE once per epoch
    criterion = HardNegativeNCE(alpha=1.0, beta=0.5).to(device)
    temp = 0.07
    dtype = model.module.dtype

    with torch.set_grad_enabled(training):
        for batch in loader:
            start = time.time()
            inputs, target_emb = batch
            inputs = inputs.to(device)
            target_emb = target_emb.to(device)
            print(inputs.keys())

            # forward: your model now takes both video paths and edits
            inf_start = time.time()
            query_emb = model(inputs)     # (B, D)
            inf_end = time.time()
            print(f"inference time = {(inf_end - inf_start):.2f}s")

            target_emb = target_emb.to(dtype=dtype)

            query_emb  = F.normalize(query_emb, dim=1)   # (B, D) → unit‐length rows
            target_emb = F.normalize(target_emb, dim=1)

            # print(f"Q, {query_emb.min().item() = }, {query_emb.max().item() = }, {query_emb.isnan().any().item() = }")
            # print(f"T, {target_emb.min().item() = }, {target_emb.max().item() = }, {target_emb.isnan().any().item() = }")    

            # Hard‐Negative NCE loss
            loss = criterion(query_emb, target_emb, temp)
            print(f"{loss.item() = }")

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            bs = target_emb.size(0) # bs = batch_size
            total_loss    += loss.item() * bs
            total_samples += bs

            end = time.time()
            
            print(f"time per batch = {(end - start):.2f}s")

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    return avg_loss


def train(model,
          train_loader: VideoPathsDataLoader,
          val_loader: VideoPathsDataLoader,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          rank):
    best_r1    = 0.0
    best_state = None
    patience   = 100
    no_improve = 0
    logs       = []

    if rank == 0:
        now       = datetime.datetime.now()
        stamp     = now.strftime("%Y%m%d_%H%M%S")   # e.g. "20250506_144816"
        base_dir  = "checkpoints"
        run_dir   = os.path.join(base_dir, stamp)
        os.makedirs(run_dir, exist_ok=True)

    # print("Profiling first batch of training…")

    # one_epoch_profiled(Phase.TRAINING, model, train_loader, device="cuda",
    #       optimizer=optimizer, do_profile=True)

    for epoch in range(epochs):
        if rank == 0:
            print(f"\n=== Epoch {epoch} ===")

        # 1) train
        train_loss = one_epoch(
            Phase.TRAINING,
            model,
            train_loader,
            optimizer
        )
        if rank == 0:
            print(f"  TRAIN   loss = {train_loss:.4f}")

        # 2) validation‐loss (no grads)
        # val_loss = one_epoch(
        #     Phase.VALIDATION,
        #     model,
        #     val_loader,
        #     device,
        #     optimizer=None
        # )
        # print(f"  VAL     loss = {val_loss:.4f}")

        # 3) retrieval metrics on the full val set
        recaller = RecallAtK(
            csv_file    = val_loader.dataset.csv_file,
            model       = model,
            batch_size  = val_loader.loader.batch_size,
            device      = model.module.device,
            num_workers = val_loader.loader.num_workers
        )
        rec = recaller.evaluate()
        print("  VAL   Recall@1,5,10,50 = "
              f"{rec['R@1']:.4f}, {rec['R@5']:.4f}, "
              f"{rec['R@10']:.4f}, {rec['R@50']:.4f}")

        logs.append({
            'train_loss': train_loss,
            # 'val_loss':   val_loss,
            'recall':     rec,
        })

        # 4) checkpoint on best R@1
        current_r1 = rec['R@1']
        if current_r1 >= best_r1 and rank == 0:
            best_r1    = current_r1
            no_improve = 0
            best_state = {
                'epoch':           epoch,
                'model_state':     model.module.head.state_dict(), # only save head
                'optimizer_state': optimizer.state_dict(),
                'train_loss':      train_loss,
                # 'val_loss':        val_loss,
                'recall':          rec,
            }
            filename = f"{model.module.head_name}checkpoint_best_r1={best_r1:.4f}_epoch={epoch}.pt"
            path = os.path.join(run_dir, filename)
            torch.save(
                best_state,
                path
            )
            print(f"  *** New best R@1 = {best_r1:.4f}, saved checkpoint.")
        else:
            no_improve += 1

        # 5) error/early‐stop
        if math.isnan(train_loss) or math.isinf(train_loss):
            print("Stopping—train loss is invalid.")
            break
        if no_improve >= patience:
            print(f"No improvement in R@1 for {patience} epochs. Early stopping.")
            break

    return best_state, logs