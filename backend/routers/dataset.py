import time
import os
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse

DATA_DIR = Path(__file__).parent.parent / "data"
router = APIRouter()


@router.post("/dataset/upload")
async def upload(label: str = Form(...), file: UploadFile = File(...)):
    label = label.strip().lower().replace(" ", "_")
    if not label:
        raise HTTPException(status_code=400, detail="Label required")

    label_dir = DATA_DIR / label
    label_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{int(time.time() * 1000)}.webm"
    (label_dir / filename).write_bytes(await file.read())

    count = len(list(label_dir.glob("*.webm")))
    return {"label": label, "file": filename, "total_in_class": count}


@router.get("/dataset/stats")
async def stats():
    if not DATA_DIR.exists():
        return {"classes": {}, "total": 0}

    classes = {}
    for d in sorted(DATA_DIR.iterdir()):
        if d.is_dir():
            classes[d.name] = len(list(d.glob("*.webm")))

    return {"classes": classes, "total": sum(classes.values())}


@router.get("/dataset/files/{label}")
async def list_files(label: str):
    label = label.strip().lower().replace(" ", "_")
    label_dir = DATA_DIR / label
    if not label_dir.exists():
        return {"label": label, "files": []}

    files = []
    for f in sorted(label_dir.glob("*.webm"), key=lambda x: x.stat().st_mtime, reverse=True):
        stat = f.stat()
        files.append({
            "name": f.name,
            "size": stat.st_size,
            "created_at": int(stat.st_mtime * 1000),
        })
    return {"label": label, "files": files}


@router.get("/dataset/audio/{label}/{filename}")
async def serve_audio(label: str, filename: str):
    label = label.strip().lower().replace(" ", "_")
    filepath = DATA_DIR / label / filename
    if not filepath.exists() or not filepath.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(filepath, media_type="audio/webm")
