import torch
from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
from app.model.bert import LogBERT

model, miner, device = None, None, None
DIST_ID, MASK_ID = 0, -1

def initialize_model(cfg):
    global model, miner, device, MASK_ID

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ Properly load Drain3 miner with trained state
    miner = TemplateMiner(persistence_handler=FilePersistence(cfg.drain_state))
    miner.drain.add_catch_all_cluster = False  # ✅ Prevent creating new templates

    # ✅ Load trained model
    model = LogBERT(vocab_size=cfg.vocab_size).to(device)
    state = torch.load(cfg.checkpoint, map_location=device)
    model.load_state_dict(state.get("model", state), strict=True)
    model.eval()

    MASK_ID = cfg.vocab_size - 1