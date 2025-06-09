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

if __name__ == "__main__":

    args = args_parse()
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

    device = "cuda"

    model = Encoder(model_name=vlm, head = "MLP", autoregressive=mode)
    optim = torch.optim.Adam(model.head.parameters())


    # prof = cProfile.Profile()
    # prof.enable()
    best_state, logs = train(model, training_loader, validation_loader, optim, device=device, epochs=epochs)
    # prof.disable()
    # stats = pstats.Stats(prof).sort_stats("cumtime")
    # stats.print_stats(20)

    print(logs)
