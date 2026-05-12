import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from collections import defaultdict

from ml.features import audio_bytes_to_wav_array, extract_mel, chunk_audio, augment_mel, normalize_mel
from ml.model import build_model

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_PATH = Path(__file__).parent.parent / "model.pth"
LABELS_PATH = Path(__file__).parent.parent / "labels.json"


class AudioDataset(Dataset):
    def __init__(self, samples: list[tuple[np.ndarray, int]], mean: float, std: float, augment: bool = False):
        self.samples = samples
        self.mean = mean
        self.std = std
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mel, label = self.samples[idx]
        mel = normalize_mel(mel, self.mean, self.std)
        if self.augment:
            mel = augment_mel(mel)
        return torch.tensor(mel).unsqueeze(0), label


def load_labels() -> list[str]:
    if LABELS_PATH.exists():
        return json.loads(LABELS_PATH.read_text())
    return sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])


def save_labels(labels: list[str]):
    LABELS_PATH.write_text(json.dumps(labels))


def collect_samples(labels: list[str]) -> list[tuple[np.ndarray, int]]:
    samples = []
    for idx, label in enumerate(labels):
        label_dir = DATA_DIR / label
        if not label_dir.exists():
            continue
        for audio_file in label_dir.glob("*.webm"):
            try:
                raw = audio_file.read_bytes()
                wav = audio_bytes_to_wav_array(raw)
                for chunk in chunk_audio(wav):
                    mel = extract_mel(chunk)
                    samples.append((mel, idx))
            except Exception:
                pass
    return samples


def compute_norm_stats(samples: list[tuple[np.ndarray, int]]) -> tuple[float, float]:
    all_mels = np.stack([mel for mel, _ in samples])
    return float(all_mels.mean()), float(all_mels.std())


def stratified_split(samples: list[tuple[np.ndarray, int]], val_ratio: float = 0.2):
    """Stratified train/val split. Classes with <4 samples go entirely to train."""
    by_class: dict[int, list] = defaultdict(list)
    for s in samples:
        by_class[s[1]].append(s)

    train, val = [], []
    for class_samples in by_class.values():
        if len(class_samples) < 4:
            train.extend(class_samples)
        else:
            n_val = max(1, int(len(class_samples) * val_ratio))
            # shuffle within class
            perm = np.random.permutation(len(class_samples)).tolist()
            idxs = [class_samples[i] for i in perm]
            val.extend(idxs[:n_val])
            train.extend(idxs[n_val:])

    return train, val


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            total_loss += criterion(out, y).item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
    model.train()
    return (total_loss / total if total else 0.0), (correct / total if total else 0.0)


def train(progress_state: dict, epochs: int = 30, batch_size: int = 16, lr: float = 1e-3):
    labels = load_labels()
    num_classes = len(labels)

    if num_classes < 2:
        progress_state.update({"status": "error", "message": "En az 2 kategori gerekli"})
        return

    all_samples = collect_samples(labels)
    if not all_samples:
        progress_state.update({"status": "error", "message": "Hiç ses bulunamadı"})
        return

    # Normalization stats from full dataset
    mel_mean, mel_std = compute_norm_stats(all_samples)

    # Train/val split
    use_val = len(all_samples) >= 8
    if use_val:
        train_samples, val_samples = stratified_split(all_samples, val_ratio=0.2)
    else:
        train_samples, val_samples = all_samples, []

    effective_batch = min(batch_size, len(train_samples))

    train_ds = AudioDataset(train_samples, mel_mean, mel_std, augment=True)
    train_loader = DataLoader(train_ds, batch_size=effective_batch, shuffle=True, drop_last=False)

    val_loader = None
    if val_samples:
        val_ds = AudioDataset(val_samples, mel_mean, mel_std, augment=False)
        val_loader = DataLoader(val_ds, batch_size=effective_batch, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 100)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    progress_state.update({
        "status": "running",
        "epoch": 0,
        "total_epochs": epochs,
        "accuracy": 0.0,
        "val_accuracy": None,
    })

    best_score = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        correct = total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)

        scheduler.step()
        train_acc = correct / total if total else 0.0

        val_acc = None
        if val_loader:
            _, val_acc = evaluate(model, val_loader, criterion, device)

        # Track best model by val_acc if available, else train_acc
        score = val_acc if val_acc is not None else train_acc
        if score >= best_score:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        progress_state.update({
            "epoch": epoch,
            "accuracy": round(train_acc, 4),
            "val_accuracy": round(val_acc, 4) if val_acc is not None else None,
        })

    # Save best model
    state_to_save = best_state if best_state is not None else model.state_dict()
    torch.save({
        "state_dict": state_to_save,
        "labels": labels,
        "num_classes": num_classes,
        "mel_mean": mel_mean,
        "mel_std": mel_std,
    }, MODEL_PATH)

    final_acc = progress_state.get("accuracy", 0.0)
    progress_state.update({
        "status": "done",
        "epoch": epochs,
        "accuracy": final_acc,
        "val_accuracy": progress_state.get("val_accuracy"),
    })
