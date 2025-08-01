import sounddevice as sd
import numpy as np
import tensorflow as tf
import torchaudio
import mido
from pathlib import Path
import json

# Path to your JSON file
config_file = 'config.json'

# Open and read the JSON file
with open(config_file, 'r') as file:
    config = json.load(file)

# --- Devices ---
devices = sd.query_devices()
for i, d in enumerate(devices):
    print(f"{i}: {d['name']} | Input Channels: {d['max_input_channels']}, Output Channels: {d['max_output_channels']}")

# --- Paths ---
YAMNET_MODEL_DIR = Path("../model/yamnet-tensorflow2-yamnet-v1")
REGRESSOR_PATH = Path("regressor_model.keras")

# --- Audio & MIDI Config ---
SAMPLE_RATE = 16000
CHANNELS = config['NUM_FX_CHANNELS'] + 1  # +1 for dry channel
BLOCK_DURATION = 1.0
FRAME_SIZE = int(SAMPLE_RATE * BLOCK_DURATION)
DEVICE_INDEX = config['DEVICE_INDEX']

# --- MIDI Config ---
port_name = config['MIDI_OUT_PORT']
inport_name = config['MIDI_IN_PORT']
outport = mido.open_output(port_name)
inport = mido.open_input(inport_name)
NUM_FX_CHANNELS = config["NUM_FX_CHANNELS"]
cc_number_FX1 = config["CC_FX_1"]
cc_number_FX2 = config["CC_FX_2"]
cc_number_FX3 = config["CC_FX_3"]
cc_fx = [cc_number_FX1, cc_number_FX2, cc_number_FX3]
cc_dry = config["CC_DRY"]
channel = config["MIDI_OUT_CHANNEL"]

# --- Relaxation input from CC22 ---
relaxation_cc = 0  # updated from MIDI in real time

# --- Gain smoothing ---
smoothed_fx_val = 64
smoothed_dry_val = 64
SMOOTH_ALPHA = 0.2  # between 0 (no change) and 1 (instant jump)

# --- Load Models ---
yamnet_model = tf.saved_model.load(str(YAMNET_MODEL_DIR))
regressor = tf.keras.models.load_model(REGRESSOR_PATH)

# --- Helpers ---
def predict_val_ar(audio_np):
    tensor = tf.convert_to_tensor(audio_np, dtype=tf.float32)
    _, embeddings, _ = yamnet_model(tensor)
    embedding = tf.reduce_mean(embeddings, axis=0)
    va = regressor(tf.expand_dims(embedding, axis=0)).numpy()[0]
    return va

def smooth(old_val, new_val, alpha=SMOOTH_ALPHA):
    return old_val * (1 - alpha) + new_val * alpha


def callback(indata, frames, time_info, status):
    global smoothed_fx_val, smoothed_dry_val

    if status:
        print(f"⚠️ {status}")

    # Normalize all channels
    indata = indata.astype(np.float32)
    indata /= np.max(np.abs(indata)) + 1e-8

    # Predict for each FX channel
    fx_va = [predict_val_ar(indata[:, i]) for i in range(NUM_FX_CHANNELS)]
    dry_va = predict_val_ar(indata[:, -1])

    relaxation_norm = relaxation_cc / 127.0

    min_val = 60
    max_val = 90

    # --- Compute FX gains based on comparison with Dry ---
    target_fx_vals = []
    for i, va in enumerate(fx_va):
        if va[1] > dry_va[1]:
            # FX channel is more arousing → attenuate
            fx_val = int(min_val + (1.0 - relaxation_norm) * (max_val - min_val))
        else:
            # FX channel is less arousing → boost
            fx_val = 127 - int(min_val + (1.0 - relaxation_norm) * (max_val - min_val))
        target_fx_vals.append(fx_val)

    # --- Compute Dry gain based on comparison with average FX arousal ---
    avg_fx_arousal = np.mean([va[1] for va in fx_va])
    if dry_va[1] > avg_fx_arousal:
        target_dry_val = int(min_val + (1.0 - relaxation_norm) * (max_val - min_val))
    else:
        target_dry_val = 127 - int(min_val + (1.0 - relaxation_norm) * (max_val - min_val))


    # Smooth and send
    for i in range(NUM_FX_CHANNELS):
        smoothed_fx_val = smooth(smoothed_fx_val, target_fx_vals[i])
        fx_val = int(np.clip(round(smoothed_fx_val), 0, 127))
        msg = mido.Message('control_change', control=cc_fx[i], value=fx_val, channel=channel)
        outport.send(msg)
        print('sent ',msg)

    smoothed_dry_val = smooth(smoothed_dry_val, target_dry_val)
    dry_val = int(np.clip(round(smoothed_dry_val), 0, 127))
    msg = mido.Message('control_change', control=cc_dry, value=dry_val, channel=channel)
    outport.send(msg)
    print('sent ',msg)


    # Log
    fx_str = " | ".join([f"FX{i+1}: V={va[0]:.2f}, A={va[1]:.2f}" for i, va in enumerate(fx_va)])
    print(f"🔊 {fx_str} | Dry: V={dry_va[0]:.2f}, A={dry_va[1]:.2f} | Relax={relaxation_norm:.2f} | Mix: Dry={dry_val}")


# --- Main loop ---
with sd.InputStream(
    device=DEVICE_INDEX,
    channels=CHANNELS,
    samplerate=SAMPLE_RATE,
    blocksize=FRAME_SIZE,
    callback=callback
):
    print("🎧 Realtime valence/arousal + EEG relaxation mixing (smoothed)...")
    while True:
        sd.sleep(50)

        # --- MIDI input processing ---
        for msg in inport.iter_pending():
            print("RECEIVED MSG",msg)
            if msg.type == 'control_change':
                if msg.control == 22:
                    relaxation_cc = msg.value
                elif msg.control == 23:
                    pass  # attention not used yet
