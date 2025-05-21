import torch
import torch.nn.functional as F
from dataset import VideoPathsDataLoader

class RecallAtK:
    """
    Compute Recall@1,5,10,50 for retrieval of target embeddings
    given a model that maps (video_paths, edits) -> query embeddings.
    """
    def __init__(
        self,
        csv_file: str,
        model: torch.nn.Module,
        batch_size: int = 32,
        device: str = "cuda",
        num_workers: int = 0,
        dtype: torch.dtype = torch.bfloat16
    ):
        self.device = device
        self.model = model.to(device).eval()
        # our custom loader returns dicts with video_paths, embs, edits
        self.loader = VideoPathsDataLoader(
            csv_file,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        self.dtype = dtype

    def _compute_queries_and_targets(self):
        qs = []
        ts = []
        with torch.no_grad():
            for batch in self.loader:
                inputs, target_emb = batch
                # model returns (B, D) query embeddings
                q = self.model(inputs).to(self.device)
                t = target_emb.to(self.device, dtype=self.dtype)
                qs.append(q)
                ts.append(t)
        queries = torch.cat(qs, dim=0)  # (N, D)
        targets = torch.cat(ts, dim=0)  # (N, D)
        return queries, targets

    def _compute_recalls(
        self,
        queries: torch.Tensor,
        targets: torch.Tensor,
        ks=(1, 5, 10, 50),
    ) -> dict:
        # normalize
        qn = F.normalize(queries, dim=1)
        tn = F.normalize(targets, dim=1)
        sim = qn @ tn.T  # (N, N)
        N = sim.size(0)
        idxs = torch.arange(N, device=self.device)
        recalls = {}
        for k in ks:
            k_ = min(k, N)
            topk = sim.topk(k_, dim=1).indices  # (N, k_)
            hits = (topk == idxs.unsqueeze(1)).any(dim=1).float()
            recalls[f"R@{k_}"] = hits.mean().item()
        return recalls

    def evaluate(self) -> dict:
        """
        Returns a dict:
          { "R@1": float, "R@5": float, "R@10": float, "R@50": float }
        """
        queries, targets = self._compute_queries_and_targets()
        return self._compute_recalls(queries, targets)