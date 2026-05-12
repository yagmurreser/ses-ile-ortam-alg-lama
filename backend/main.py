from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pathlib import Path

from routers import inference, dataset, training
from routers.training import _load_model_into_app


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model_into_app(app)
    yield


app = FastAPI(title="Ses Sınıflandırma API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(inference.router)
app.include_router(dataset.router)
app.include_router(training.router)


@app.get("/health")
def health():
    return {"status": "ok"}
