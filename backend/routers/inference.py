from fastapi import APIRouter, UploadFile, File, HTTPException, Request
import torch
import torch.nn.functional as F

from ml.features import audio_bytes_to_tensors

router = APIRouter()

UNKNOWN_THRESHOLD = 0.5


@router.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    model_data = getattr(request.app.state, "model_data", None)
    if model_data is None:
        raise HTTPException(status_code=503, detail="Model yüklü değil. Önce eğit.")

    model = model_data["model"]
    labels = model_data["labels"]
    device = model_data["device"]
    mel_mean = model_data.get("mel_mean", 0.0)
    mel_std = model_data.get("mel_std", 1.0)

    audio_bytes = await file.read()
    try:
        tensors = audio_bytes_to_tensors(audio_bytes, mel_mean=mel_mean, mel_std=mel_std)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ses işleme hatası: {e}")

    model.eval()
    all_probs = []
    with torch.no_grad():
        for t in tensors:
            out = model(t.to(device))
            probs = F.softmax(out, dim=1).cpu().squeeze(0)
            all_probs.append(probs)

    avg_probs = torch.stack(all_probs).mean(dim=0)
    pred_idx = avg_probs.argmax().item()
    confidence = avg_probs[pred_idx].item()

    predicted_label = labels[pred_idx] if confidence >= UNKNOWN_THRESHOLD else "bilinmeyen"

    return {
        "label": predicted_label,
        "confidence": round(confidence, 4),
        "all_scores": {labels[i]: round(avg_probs[i].item(), 4) for i in range(len(labels))},
        "chunks_processed": len(tensors),
    }
