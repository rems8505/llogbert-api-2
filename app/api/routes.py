from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List
import torch

from app.config import settings
from app.model.inference import model, miner, device, MASK_ID, DIST_ID
from app.parser.validator import validate_line
from app.model.bert import miss_count

router = APIRouter()

class LogBatch(BaseModel):
    lines: List[str] = Field(..., min_items=32, max_items=32)

@router.post("/score_batch")
def score_batch(batch: LogBatch):
    ids = []
    for line in batch.lines:
        try:
            clean_line = validate_line(line)
            cid = miner.add_log_message(clean_line)["cluster_id"]
            if cid >= MASK_ID:
                return {"is_anomaly": True, "reason": "unseen_template", "cluster_id": int(cid), "score": 1.0}
            ids.append(cid + 1)
        except ValueError as ve:
            raise HTTPException(status_code=422, detail=str(ve))

    seq = torch.tensor([[DIST_ID] + ids], dtype=torch.long, device=device)
    miss_cnt, num_mask = miss_count(model, seq, settings.topk, settings.mask_ratio)
    return {
        "is_anomaly": miss_cnt > settings.r,
        "reason": "miss_ratio",
        "miss_count": miss_cnt,
        "masked": num_mask,
        "g": settings.topk,
        "r": settings.r,
        "score": round(miss_cnt / num_mask, 4)
    }
