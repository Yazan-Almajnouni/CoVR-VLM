# run once, e.g. at the top of run.py
import torch
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
device = "cuda"

from models import Encoder
from loss import HardNegativeNCE
from dataset import VideoPathsDataLoader


def report_mem(label=""):
    print(f"\n— {label} —")
    print("  allocated:", 
          torch.cuda.memory_allocated() / 1e9, "GB")
    print("  reserved: ", 
          torch.cuda.memory_reserved()  / 1e9, "GB")
    print("  peak alloc:", 
          torch.cuda.max_memory_allocated() / 1e9, "GB")
    print("  peak resv: ", 
          torch.cuda.max_memory_reserved()  / 1e9, "GB")
    print(torch.cuda.memory_summary())

# build model and optimizer
autoregressive = False
model_name="Qwen/Qwen2.5-VL-3B-Instruct"
batch_size = 208
print(f"{autoregressive = }")
print(f"{model_name = }")
print(f"{batch_size = }")
model = Encoder(model_name=model_name, head = "MLP", autoregressive=autoregressive).to(device)
optim  = torch.optim.Adam(model.head.parameters())
report_mem("model initialized")

training_annotations = "annotations/train_set.csv" 
training_loader = VideoPathsDataLoader(training_annotations, batch_size=batch_size, num_workers=0)

# grab a single batch
dl = iter(training_loader)
(inputs, targets) = next(dl)
# send inputs & targets
inputs.to(device)
targets = targets.to(device, dtype=model.dtype)

report_mem("after data transfer")

# forward + backward
model.head.train()
query_emb = model(inputs)
report_mem("after fwd")
loss = HardNegativeNCE()(query_emb, targets, temp=0.07)
loss.backward()
report_mem("after bwd")
