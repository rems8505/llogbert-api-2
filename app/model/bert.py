# src/fastapi/logbert_api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List
import torch
import html
import re

from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence



import torch
import torch.nn as nn
import torch.nn.functional as F

class LogBERT(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 128, max_len: int = 33, nhead: int = 4, nlayers: int = 2):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_len, dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=nhead,
            dim_feedforward=256,
            activation='relu',
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.drop = nn.Dropout(0.1)
        self.mlkp_head = nn.Linear(dim, vocab_size)  # ← FIXED name here

    def forward(self, x):
        B, L = x.size()
        tok = self.token_emb(x)
        pos = self.pos_emb(torch.arange(L, device=x.device))
        z = tok + pos
        z = self.drop(z)
        z = self.encoder(z)
        return self.mlkp_head(z), z  # ← Also changed here


# ---------------------- Miss Count Function ----------------------
IGNORE_IDX = -100

@torch.no_grad()
def miss_count(model: LogBERT,
               seq: torch.Tensor,
               topk: int = 5,
               mask_ratio: float = 0.3):
    """Return (miss_count, num_masked) for one sequence [1,L] (DIST at pos0)."""
    L = seq.size(1)
    num_mask = max(1, int((L - 1) * mask_ratio))
    pos = torch.randperm(L - 1, device=seq.device)[:num_mask] + 1  # skip DIST
    labels = torch.full_like(seq, IGNORE_IDX)
    labels[0, pos] = seq[0, pos]
    seq[0, pos] = model.token_emb.num_embeddings - 1               # MASK id
    logits, _ = model(seq)
    top = logits.topk(k=topk, dim=-1).indices
    hit = top.eq(labels.unsqueeze(-1)).any(-1)
    miss = (~hit)[0, pos]
    return int(miss.sum()), num_mask



# --------------------------- Input Model ---------------------------
class LogBatch(BaseModel):
    lines: List[str] = Field(..., min_items=32, max_items=32)

    @validator("lines", each_item=True)
    def validate_and_escape(cls, line: str) -> str:
        stripped_line = line.strip()
        if not LOG_REGEX.match(stripped_line):
            raise ValueError(f"Invalid OpenStack log format: {stripped_line}")
        return html.escape(stripped_line)


# -------------------------- Create App ----------------------------
def create_app(
    checkpoint_path: str,
    drain_state_path: str,
    vocab_size: int,
    topk: int = 5,
    r: int = 3,
    mask_ratio: float = 0.3
) -> FastAPI:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Drain3 miner
    miner = TemplateMiner(persistence_handler=FilePersistence(drain_state_path))
    miner.drain.add_catch_all_cluster = False  # freeze

    # Model setup
    model = LogBERT(vocab_size=vocab_size).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state.get("model", state), strict=True)
    model.eval()

    DIST_ID = 0
    MASK_ID = vocab_size - 1

    app = FastAPI(title="LogBERT Batch Inference")

    @app.post("/score_batch")
    def score_batch(batch: LogBatch):
        ids = []
        for line in batch.lines:
            result = miner.add_log_message(line)
            cid = result["cluster_id"]
            if cid >= MASK_ID:
                return {
                    "is_anomaly": True,
                    "reason": "unseen_template",
                    "cluster_id": int(cid),
                    "score": 1.0
                }
            ids.append(cid + 1)

        seq = torch.tensor([[DIST_ID] + ids], dtype=torch.long, device=device)
        miss_cnt, num_mask = miss_count(model, seq, topk=topk, mask_ratio=mask_ratio)
        is_anom = miss_cnt > r
        return {
            "is_anomaly": bool(is_anom),
            "reason": "miss_ratio",
            "miss_count": miss_cnt,
            "masked": num_mask,
            "g": topk,
            "r": r,
            "score": round(miss_cnt / num_mask, 4)
        }

    @app.get("/")
    def root():
        return {"msg": "POST JSON {'lines': [32 raw lines]} to /score_batch"}

    return app
