from trainer import train
from dataset import VideoPathsDataLoader
from models import Encoder
import torch
import torch.nn as nn
import cProfile, pstats
import argparse

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
    args = parser.parse_args()

    return args

if __name__ == "__main__":

    args = args_parse()
    batch_size = args.batch
    vlm = args.vlm

    torch.manual_seed(0)
    training_annotations = "annotations/train_set.csv" 
    validation_annotations = "annotations/mini_val_set.csv"
    training_loader = VideoPathsDataLoader(training_annotations, batch_size=batch_size)
    validation_loader = VideoPathsDataLoader(validation_annotations, batch_size=batch_size)

    device = "cuda"

    model = Encoder(model_name=vlm, head = "MLP")
    optim = torch.optim.Adam(model.head.parameters())


    # prof = cProfile.Profile()
    # prof.enable()
    best_state, logs = train(model, training_loader, validation_loader, optim, device=device, epochs=1)
    # prof.disable()
    # stats = pstats.Stats(prof).sort_stats("cumtime")
    # stats.print_stats(20)

    print(logs)
