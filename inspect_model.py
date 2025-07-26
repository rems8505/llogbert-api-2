import torch

path = "./app/model/best_ft.pt"  # or full path if needed

# Load only metadata
state = torch.load(path, map_location="cpu")

# If saved with "model" key (common in training checkpoints)
if "model" in state:
    model_state = state["model"]
else:
    model_state = state

print("\n=== Parameter Names and Shapes Preview ===")
for i, (k, v) in enumerate(model_state.items()):
    print(f"{i+1:03d}: {k} => shape: {tuple(v.shape)}")
    if i >= 20:
        break

print(f"\nTotal parameters: {len(model_state)}")
