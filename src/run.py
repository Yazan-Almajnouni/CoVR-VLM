from trainer import train
from dataset import VideoPathsDataLoader
from models import Encoder
import torch
import torch.nn as nn
import cProfile, pstats
import argparse

from torch.profiler import profile, record_function, ProfilerActivity

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

import time

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--batch",
        type=int,
        default=32,
        help="batch size"
    )
    parser.add_argument(
        "-v", "--vlm",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="VLM to use"
    )

    parser.add_argument(
        "-e", "--epochs",
        type=int,
        default=10,
        help="number of epochs"
    )

    parser.add_argument(
        "-d", "--dataset",
        type=str,
        default="test",
        help="dataset: mini, test, val, train"
    )

    parser.add_argument(
        "-m", "--mode",
        type=str,
        default="y",
        help="VLM mode. Autoregressive if y"
    )

    args = parser.parse_args()

    return args

def main(rank, world_size, args):
    ddp_setup(rank, world_size)

    batch_size = args.batch
    vlm = args.vlm
    epochs = args.epochs
    dataset = args.dataset
    mode = args.mode
    if mode == "y":
        mode = True
    elif mode == "n":
        mode = False
    else:
        raise ValueError
    print(f"{mode = }")
    print(f"{type(mode) = }")

    if dataset == "mini":
        training_annotations = "annotations/mini_train_set.csv" 
        validation_annotations = "annotations/mini_val_set.csv"
    elif dataset == "test":
        training_annotations = "annotations/train_set.csv" 
        validation_annotations = "annotations/val_set.csv"
    else:
        raise ValueError       


    torch.manual_seed(0)
    training_loader = VideoPathsDataLoader(training_annotations, batch_size=batch_size)
    validation_loader = VideoPathsDataLoader(validation_annotations, batch_size=batch_size)

    model = Encoder(model_name=vlm, head = "MLP", autoregressive=mode, rank=rank)
    model = DDP(model, device_ids=[rank])
    optim = torch.optim.Adam(model.module.head.parameters())



    # prof = cProfile.Profile()
    # prof.enable()
    best_state, logs = train(model, training_loader, validation_loader, optim, epochs=epochs, rank=rank)
    # prof.disable()
    # stats = pstats.Stats(prof).sort_stats("cumtime")
    # stats.print_stats(20)

    print(logs)

    destroy_process_group()


if __name__ == "__main__":
    args = args_parse()
    world_size = torch.cuda.device_count()
    print(f"{world_size = }")
    start = time.time()
    mp.spawn(main, args=(world_size, args), nprocs=world_size)
    end = time.time()
    
    print(f"total training time = {(end - start):.2f}")

