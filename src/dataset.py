import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

class VideoPathsDataset(Dataset):
    """
    Expects a CSV with columns: pth1, pth2, edit.
    Appends .mp4 to pth1, loads .pth from pth2, and returns the emb tensor.
    """
    def __init__(self, csv_file: str):
        self.df = pd.read_csv(csv_file)
        self.base_path = (
            "/ibex/user/majnouym/ivul_research/"
            "cpvr_challenge/CoVR-VidLLM-CVPRW25/datasets/WebVid/8M/"
        )
        self.csv_file = csv_file

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        video_path = os.path.join(
            self.base_path,
            "train",
            f"{row['pth1']}.mp4"
        )

        emb_path = os.path.join(
            self.base_path,
            "blip-vid-embs-large-all",
            f"{row['pth2']}.pth"
        )

        # load the .pth embedding (assumes it was saved with torch.save)
        emb_tensor = torch.load(emb_path) # shape = (F, D)
        # if you need float tensors, uncomment:
        # emb_tensor = emb_tensor.float()

        return {
            "video_path": video_path,
            "emb":         emb_tensor,
            "edit":        row["edit"],
        }
    

class VideoPathsDataLoader:
    """
    Wraps torch.utils.data.DataLoader to batch video paths, emb tensors & edits.
    """
    def __init__(
        self,
        csv_file: str,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 4,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        prompt: str = "{}. Describe the changed video in details as if you have seen it.",
    ):
        self.dataset = VideoPathsDataset(csv_file)
        self.loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn
        )

        self.csv_file = csv_file
        self.model_name = model_name
        self.prompt = prompt

        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)

        self.processor.tokenizer.padding_side = "left" # force left padding on the tokenizer to satisfy flash‐attn causalmask


    def _collate_fn(self, batch: list) -> tuple:
        """
        batch is a list of dicts; we turn it to tuple: (vlm_inputs, target_embeddings)
        """
        embs = []
        conversations = []

        for item in batch:
            path = item['video_path']
            emb = item['emb']
            edit = item['edit']

            embs.append(emb)

            formatted_prompt = self.prompt.format(edit)

            user_content = [
                {"type": "text", "text": formatted_prompt},
                {
                    "video": path,
                    "total_pixels": 20480 * 28 * 28,
                    "min_pixels": 16 * 28 * 28,
                    "nframes": 15,
                },
            ]

            conversation = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",   "content": user_content},
            ]

            conversations.append(conversation)

        text = self.processor.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=False
        )

        image_inputs, video_inputs, vk = process_vision_info(conversations, return_video_kwargs=True)
        

        fps = vk["fps"]

        inputs = self.processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            fps=fps,
            padding=True,
            return_tensors="pt",
        )


        embs = torch.stack(embs, dim=0).mean(dim=1)


        return (inputs, embs)

    def __iter__(self):
        return iter(self.loader)

    def __len__(self) -> int:
        return len(self.loader)

