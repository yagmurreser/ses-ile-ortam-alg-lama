import asyncio
from pathlib import Path
from fastapi import APIRouter, Request, HTTPException

MODEL_PATH = Path(__file__).parent.parent / "model.pth"
router = APIRouter()

_progress: dict = {"status": "idle"}


@router.post("/train")
async def start_train(request: Request, epochs: int = 30):
    global _progress
    if _progress.get("status") == "running":
        raise HTTPException(status_code=409, detail="Training already running")

    _progress = {"status": "starting", "epoch": 0, "total_epochs": epochs, "accuracy": 0.0}

    async def run():
        from ml.trainer import train
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, train, _progress, epochs)
        # reload model into app state after training
        _load_model_into_app(request.app)

    asyncio.create_task(run())
    return {"status": "started", "epochs": epochs}


@router.get("/train/status")
async def train_status():
    return dict(_progress)


@router.get("/model/info")
async def model_info(request: Request):
    model_data = getattr(request.app.state, "model_data", None)
    if model_data is None:
        return {"loaded": False, "message": "No model loaded"}
    return {
        "loaded": True,
        "labels": model_data["labels"],
        "num_classes": model_data["num_classes"],
        "device": str(model_data["device"]),
    }


def _load_model_into_app(app):
    import torch
    from ml.model import build_model

    if not MODEL_PATH.exists():
        return

    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    labels = checkpoint["labels"]
    num_classes = checkpoint["num_classes"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    app.state.model_data = {
        "model": model,
        "labels": labels,
        "num_classes": num_classes,
        "device": device,
        "mel_mean": checkpoint.get("mel_mean", 0.0),
        "mel_std": checkpoint.get("mel_std", 1.0),
    }
