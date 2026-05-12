import numpy as np
import librosa
import io
import static_ffmpeg
import shutil as _shutil

static_ffmpeg.add_paths(weak=False)
_ffmpeg_exe = _shutil.which("ffmpeg") or "ffmpeg"
_ffprobe_exe = _shutil.which("ffprobe") or _ffmpeg_exe

from pydub import AudioSegment
AudioSegment.converter = _ffmpeg_exe
AudioSegment.ffprobe = _ffprobe_exe

SR = 22050
DURATION = 5
N_MELS = 64
HOP_LENGTH = 512
N_FRAMES = 216  # 5s * 22050 / 512


def audio_bytes_to_wav_array(audio_bytes: bytes) -> np.ndarray:
    seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
    seg = seg.set_frame_rate(SR).set_channels(1)
    samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
    samples /= np.iinfo(seg.array_type).max
    return samples


def extract_mel(samples: np.ndarray) -> np.ndarray:
    target_len = SR * DURATION
    if len(samples) < target_len:
        samples = np.pad(samples, (0, target_len - len(samples)))
    else:
        samples = samples[:target_len]

    mel = librosa.feature.melspectrogram(
        y=samples, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    if mel_db.shape[1] < N_FRAMES:
        mel_db = np.pad(mel_db, ((0, 0), (0, N_FRAMES - mel_db.shape[1])))
    else:
        mel_db = mel_db[:, :N_FRAMES]

    return mel_db.astype(np.float32)


def chunk_audio(samples: np.ndarray) -> list[np.ndarray]:
    chunk_len = SR * DURATION
    chunks = []
    for start in range(0, len(samples), chunk_len):
        chunk = samples[start : start + chunk_len]
        if len(chunk) < chunk_len // 2:
            break
        chunks.append(chunk)
    return chunks if chunks else [samples]


def augment_mel(mel: np.ndarray) -> np.ndarray:
    """SpecAugment + noise + time shift. Input shape: (N_MELS, N_FRAMES)."""
    mel = mel.copy()
    n_mels, n_frames = mel.shape
    fill = float(mel.mean())

    # Time shift (±10%)
    shift = np.random.randint(-n_frames // 10, n_frames // 10 + 1)
    mel = np.roll(mel, shift, axis=1)

    # Frequency masking: mask up to 15 mel bins
    f = np.random.randint(0, min(16, n_mels // 4 + 1))
    if f > 0:
        f0 = np.random.randint(0, n_mels - f)
        mel[f0 : f0 + f, :] = fill

    # Time masking: mask up to 30 frames
    t = np.random.randint(0, min(31, n_frames // 4 + 1))
    if t > 0:
        t0 = np.random.randint(0, n_frames - t)
        mel[:, t0 : t0 + t] = fill

    # Gaussian noise
    mel += np.random.randn(*mel.shape).astype(np.float32) * 0.04

    return mel


def normalize_mel(mel: np.ndarray, mean: float, std: float) -> np.ndarray:
    return ((mel - mean) / (std + 1e-6)).astype(np.float32)


def audio_bytes_to_tensors(audio_bytes: bytes, mel_mean: float = 0.0, mel_std: float = 1.0):
    """Full pipeline: bytes → list of (1, 1, N_MELS, N_FRAMES) tensors."""
    import torch

    samples = audio_bytes_to_wav_array(audio_bytes)
    chunks = chunk_audio(samples)
    tensors = []
    for chunk in chunks:
        mel = extract_mel(chunk)
        mel = normalize_mel(mel, mel_mean, mel_std)
        t = torch.tensor(mel).unsqueeze(0).unsqueeze(0)  # (1,1,64,216)
        tensors.append(t)
    return tensors
