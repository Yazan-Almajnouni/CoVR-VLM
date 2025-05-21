from trainer import train
from dataset import VideoPathsDataLoader
from models import VLM, MLPProjector, CNNProjector, GRUProjector, Encoder
import torch
import torch.nn as nn
import cProfile, pstats
import sys

if __name__ == "__main__":
    batch_size = int(sys.argv[1])

    torch.manual_seed(0)
    training_annotations = "annotations/mini_train_set.csv" 
    validation_annotations = "annotations/mini_val_set.csv"
    training_loader = VideoPathsDataLoader(training_annotations, batch_size=batch_size)
    validation_loader = VideoPathsDataLoader(validation_annotations, batch_size=batch_size)

    device = "cuda"

    model = Encoder(head = "MLP")
    optim = torch.optim.Adam(model.head.parameters())


    # prof = cProfile.Profile()
    # prof.enable()
    best_state, logs = train(model, training_loader, validation_loader, optim, device=device, epochs=1)
    # prof.disable()
    # stats = pstats.Stats(prof).sort_stats("cumtime")
    # stats.print_stats(20)

    print(logs)
