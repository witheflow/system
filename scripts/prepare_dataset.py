import os
import numpy as np
import pandas as pd
import torchaudio
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm

# ---- Configuration ----
AUDIO_DIR = Path("../DEAM/MEMD_audio")
VALENCE_CSV = Path("../DEAM/valence.csv")
AROUSAL_CSV = Path("../DEAM/arousal.csv")
YAMNET_MODEL_DIR = Path("../model/yamnet-tensorflow2-yamnet-v1")  # directory containing saved_model.pb
OUTPUT_DIR = Path("features")
OUTPUT_DIR.mkdir(exist_ok=True)

SAMPLE_RATE = 16000
FRAME_SIZE = 15600  # YAMNet expects 0.975s = 15600 samples
STEP_SIZE = int(0.5 * SAMPLE_RATE)  # DEAM labels are every 0.5s

# ---- Load YAMNet SavedModel ----
yamnet_model = tf.saved_model.load(str(YAMNET_MODEL_DIR))

# ---- Load label data ----
val_df = pd.read_csv(VALENCE_CSV).set_index("song_id")
ar_df = pd.read_csv(AROUSAL_CSV).set_index("song_id")

# ---- Helper: run inference on one audio chunk ----
def run_yamnet(audio_np):
    tensor = tf.convert_to_tensor(audio_np, dtype=tf.float32)
    _, embeddings, _ = yamnet_model(tensor)
    return embeddings.numpy()  # shape: [N, 1024]

# ---- Main loop over songs ----
data_X = []
data_y = []

for song_id in tqdm(val_df.index):
    file_path = AUDIO_DIR / f"{int(song_id)}.mp3"
    if not file_path.exists():
        continue

    # Load and resample audio
    wav, sr = torchaudio.load(str(file_path))
    if sr != SAMPLE_RATE:
        wav = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(wav)
    wav = wav.mean(dim=0).numpy()  # convert to mono numpy

    # Slide over audio with 0.975s frames every 0.5s
    for start in range(0, len(wav) - FRAME_SIZE, STEP_SIZE):
        chunk = wav[start:start + FRAME_SIZE]
        if chunk.shape[0] != FRAME_SIZE:
            continue

        # Get YAMNet embedding
        embedding = run_yamnet(chunk)
        embedding = embedding.mean(axis=0)  # average over time

        # Align label
        timestamp_ms = 15000 + int(start / SAMPLE_RATE * 1000)
        col_name = f"sample_{timestamp_ms}ms"
        if col_name not in val_df.columns:
            continue

        val = val_df.loc[song_id, col_name]
        ar = ar_df.loc[song_id, col_name]
        if pd.isna(val) or pd.isna(ar):
            continue

        data_X.append(embedding)
        data_y.append([val, ar])

# ---- Save to disk ----
data_X = np.array(data_X)
data_y = np.array(data_y)
np.save(OUTPUT_DIR / "X_embeddings.npy", data_X)
np.save(OUTPUT_DIR / "y_val_ar.npy", data_y)

print("✅ Feature extraction complete.", data_X.shape, data_y.shape)
