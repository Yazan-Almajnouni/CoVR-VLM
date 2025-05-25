from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from transformers import Qwen2_5_VLForConditionalGeneration

class MLPProjector(nn.Module):
    def __init__(self, hidden_size = 3584, proj_size = 256, dtype = torch.bfloat16, device = "cuda"):
        super().__init__()

        self.hidden_size = hidden_size
        self.proj_size = proj_size
        self.dtype = dtype
        self.device = device

        self.model = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, proj_size)
        ).to(device = device, dtype = dtype)

    def forward(self, vlm_output):
        return self.model(vlm_output.mean(dim=1))
    
class CNNProjector(nn.Module):
    def __init__(self, hidden_size = 3584, proj_size = 256, dtype = torch.bfloat16, device = "cuda", kernel_size=3):
        super(CNNProjector, self).__init__()
        # 1D conv over the sequence dimension

        self.hidden_size = hidden_size
        self.proj_size = proj_size
        self.dtype = dtype
        self.device = device

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=hidden_size, out_channels=proj_size, kernel_size=kernel_size, padding=kernel_size // 2)
        ).to(device = device, dtype = dtype)
        # adaptive pooling to length=1
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, vlm_output):
        # x: (B, S, D)
        vlm_output = vlm_output.permute(0, 2, 1)          # -> (B, D, S)
        vlm_output = self.conv(vlm_output)                # -> (B, D_out, S)
        vlm_output = self.pool(vlm_output)                # -> (B, D_out, 1)
        return vlm_output.squeeze(-1)                     # -> (B, D_out)
    
class GRUProjector(nn.Module):
    def __init__(self, hidden_size = 3584, proj_size = 256, dtype = torch.bfloat16, device = "cuda"):

        self.hidden_size = hidden_size
        self.proj_size = proj_size
        self.dtype = dtype
        self.device = device

        self.model = nn.GRU(
            input_size=hidden_size,
            hidden_size=proj_size,
            batch_first=True,
            device=device,
            dtype=dtype
        )

    
    def forward(self, vlm_outputs):
        out, h_n = self.model(vlm_outputs)
        return h_n


class VLM(nn.Module):
    """
    loads and freezes a VLM
    """
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        dtype: torch.dtype = torch.bfloat16,
        attn_impl: str = "flash_attention_2",
        device_map: str = "auto",
    ):
        super().__init__()

        self.model_name = model_name
        self.dtype = dtype

        # load & freeze Qwen
        self.qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            attn_implementation=attn_impl,
            device_map=device_map,
        )
        for p in self.qwen.parameters():
            p.requires_grad = False

        self.hidden_size = self.qwen.config.hidden_size
        self.device = self.qwen.device

        identity_head = torch.nn.Identity().to(device=self.device, dtype=self.dtype)
        self.qwen.set_output_embeddings(identity_head)



    def forward(self, inputs) -> torch.Tensor:

        # forward through Qwen
        with torch.no_grad():
            outputs = self.qwen(
                **inputs,
                output_hidden_states=False,
                return_dict=True
            )

        last_hidden = outputs.logits  # (B, S, D)

        return last_hidden


class Encoder(nn.Module):
    def __init__(self, model_name = "Qwen/Qwen2.5-VL-7B-Instruct", head = "MLP"):
        super().__init__()
        self.vlm = VLM(model_name=model_name)
        self.vlm.eval()
        self.head = self.get_head(head)
        self.head_name = head
        self.dtype = self.head.dtype

    def forward(self, inputs):
        inputs = inputs.to(device = self.vlm.device)
        inputs = self.vlm(inputs)

        inputs.to(self.head.device)
        inputs = self.head(inputs)

        return inputs
    
    def get_head(self, head):
        heads = ["MLP", "CNN", "GRU"]
        assert head in heads, f"{head} is not in {heads}"

        hidden_size = self.vlm.hidden_size

        if head == "MLP":
            return MLPProjector(hidden_size=hidden_size)
        if head == "CNN":
            return CNNProjector(hidden_size=hidden_size)
        if head == "GRU":
            return GRUProjector(hidden_size=hidden_size)