import torch
import json
import sounddevice as sd
import numpy as np
import threading
import time
from collections import deque
import mido
import yaml
import sys
from datetime import datetime

class DualLogger:
    def __init__(self, logfile_path):
        self.terminal = sys.__stdout__
        self.logfile = open(logfile_path, "a", buffering=1)  # line-buffered

    def write(self, message):
        self.terminal.write(message)  # Print to console

        for line in message.rstrip().splitlines():
            if line.strip():
                timestamp = datetime.now().strftime("[%H:%M:%S.%f]")[:-3]  # up to milliseconds
                self.logfile.write(f"{timestamp} {line}\n")

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

sys.stdout = DualLogger("session_log.txt")




# Print MIDI and audio devices
print("Available MIDI output ports:")
for name in mido.get_output_names():
    print(name)

for idx, device in enumerate(sd.query_devices()):
    print(f"{idx}: {device['name']} ({device['max_input_channels']} in, {device['max_output_channels']} out)")

# Load config
with open('config.json', 'r') as f:
    config = json.load(f)

SAMPLE_RATE = 32000
FRAME_DURATION = 0.25
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION)
BUFFER_DURATION = 5.0
NUM_CHANNELS = config["NUM_FX_CHANNELS"] + 2  # dry + FX + full mix

# GLOBAL VARS
previous_gains = [0.0] * NUM_CHANNELS
current_stress = 0
current_attention = 0
current_strength = 127  # full strength by default
previous_dry_va = None
dry_va_history = deque(maxlen=20)  # e.g. last 10 seconds if 0.5s inference
stable_seconds = 0

# LOAD RULES
if config['SENSORS_AVAILABLE'] == "all":
    inport = mido.open_input(config['MIDI_IN_PORT'])
    rules_file = "rules_stress_attention.yaml"
elif config['SENSORS_AVAILABLE'] == "attention":
    inport = mido.open_input(config['MIDI_IN_PORT'])
    rules_file = "rules_attention.yaml"
else:
    inport = None
    rules_file = "rules_audio.yaml"

with open(rules_file, "r") as f:
    rules = yaml.safe_load(f)["rules"]

# Audio buffer class
class AudioBuffer:
    def __init__(self, max_duration_sec, sample_rate):
        self.max_samples = int(max_duration_sec * sample_rate)
        self.buffer = deque(maxlen=self.max_samples)

    def update(self, new_data):
        self.buffer.extend(new_data)

    def get_audio(self):
        arr = np.array(self.buffer, dtype=np.float32)
        if len(arr) < self.max_samples:
            arr = np.pad(arr, (self.max_samples - len(arr), 0))
        return arr

channel_buffers = [AudioBuffer(BUFFER_DURATION, SAMPLE_RATE) for _ in range(NUM_CHANNELS)]

def audio_callback(indata, frames, time, status):
    for i, buf in enumerate(channel_buffers):
        buf.update(indata[:, i])

stream = sd.InputStream(
    device=config["DEVICE_INDEX"],
    channels=NUM_CHANNELS,
    samplerate=SAMPLE_RATE,
    blocksize=FRAME_SIZE,
    callback=audio_callback
)

# Load model
model = torch.jit.load(config["MODEL_PATH"], map_location="cpu")
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)

# Inference

def compute_va_per_channel(model, channel_buffers, device="cpu"):
    model.eval()
    inputs = [
        torch.tensor(buf.get_audio(), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        for buf in channel_buffers
    ]
    batch = torch.cat(inputs, dim=0).to(device)
    with torch.no_grad():
        va = model(batch)
    return va.cpu().numpy()

# Gain functions
def boost_far_in_high_arousal_direction(dry_va, fx_vas, **kwargss):
    distances = np.linalg.norm(fx_vas - dry_va, axis=1)
    arousal_diffs = fx_vas[:, 1] - dry_va[1]
    weight = distances * (arousal_diffs > 0)
    gains = 127 * (weight / (weight.max() + 1e-6))
    return gains

def boost_far_any_direction(dry_va, fx_vas, **kwargs):
    distances = np.linalg.norm(fx_vas - dry_va, axis=1)
    gains = 127 * (distances / (distances.max() + 1e-6))
    return gains

def boost_near_only(dry_va, fx_vas, **kwargs):
    distances = np.linalg.norm(fx_vas - dry_va, axis=1)
    gains = 127 * (1 - distances / (distances.max() + 1e-6))
    return gains

def boost_near_low_arousal(dry_va, fx_vas, **kwargs):
    distances = np.linalg.norm(fx_vas - dry_va, axis=1)
    arousal_diffs = fx_vas[:, 1] - dry_va[1]
    mask = (arousal_diffs < 0)
    weight = mask * (1 - distances / (distances.max() + 1e-6))
    gains = 127 * weight
    return gains

def boost_near_audio_only(dry_va, fx_vas, **kwargs):
    distances = np.linalg.norm(fx_vas - dry_va, axis=1)
    inv_dist = 1 / (distances + 1e-6)
    gains = 127 * (inv_dist / inv_dist.max())
    return gains

def boost_far_audio_only(dry_va, fx_vas, **kwargs):
    distances = np.linalg.norm(fx_vas - dry_va, axis=1)
    gains = 127 * (distances / (distances.max() + 1e-6))
    return gains

def boost_directional_shift(dry_va, fx_vas, **kwargs):
    delta_va = kwargs.get("delta_va", None)
    if delta_va is None or np.linalg.norm(delta_va) < 1e-3:
        return boost_near_only(dry_va, fx_vas)

    delta_unit = delta_va / (np.linalg.norm(delta_va) + 1e-6)
    directions = fx_vas - dry_va
    fx_unit_dirs = directions / (np.linalg.norm(directions, axis=1, keepdims=True) + 1e-6)

    alignment = np.dot(fx_unit_dirs, delta_unit)
    gains = 127 * ((alignment + 1) / 2)
    return gains

FUNC_MAP = {
    "boost_far_in_high_arousal_direction": boost_far_in_high_arousal_direction,
    "boost_far_any_direction": boost_far_any_direction,
    "boost_near_only": boost_near_only,
    "boost_near_low_arousal": boost_near_low_arousal,
    "boost_near_audio_only": boost_near_audio_only,
    "boost_far_audio_only": boost_far_audio_only,
    "boost_directional_shift": boost_directional_shift
}

# Rule matcher
def matches_condition(rule_cond, x_dict):
    for key, cond in rule_cond.items():
        val = x_dict[key]
        if "gt" in cond and not val > cond["gt"]:
            return False
        if "lt" in cond and not val < cond["lt"]:
            return False
    return True

def get_next_mix(valence_arousal):
    global previous_dry_va, stable_seconds

    dry_va = valence_arousal[0]
    fx_vas = valence_arousal[1:-1]
    dry_va_history.append(dry_va)

    if len(dry_va_history) >= 6:
        history_array = np.stack(dry_va_history)
        recent = history_array[-3:].mean(axis=0)
        earlier = history_array[:3].mean(axis=0)
        delta_va = recent - earlier
    else:
        delta_va = np.array([0.0, 0.0])

    if len(dry_va_history) >= 5:
        history_array = np.stack(dry_va_history)
        volatility = np.std(history_array, axis=0)
        is_stable = np.all(volatility < 0.03)
    else:
        volatility = np.array([0.0, 0.0])
        is_stable = False

    stable_seconds = stable_seconds + 0.5 if is_stable else 0.0

    x_dict = {
        "delta_valence": delta_va[0],
        "delta_arousal": delta_va[1],
        "valence_volatility": volatility[0],
        "arousal_volatility": volatility[1],
        "stable_for_seconds": stable_seconds,
        "stress": current_stress / 127.0,
        "attention": current_attention / 127.0,
        "valence": dry_va[0],
        "arousal": dry_va[1]
    }

    for rule in rules:
        if matches_condition(rule.get("condition", {}), x_dict):
            func = FUNC_MAP[rule["function"]]
            fx_gains = func(dry_va, fx_vas, delta_va=delta_va)

            gains = np.array([100] + fx_gains.tolist() + [100], dtype=np.float32)
            alpha = 0.2
            smoothed = alpha * gains + (1 - alpha) * np.array(previous_gains)
            smoothed = np.clip(np.round(smoothed), 0, 127).astype(int)

            # Interpolate with strength
            strength = current_strength / 127.0
            inverted = 127 - smoothed
            final = np.round(strength * smoothed + (1 - strength) * inverted).astype(int)

            previous_gains[:] = final
            print(f"Active Rule: {rule['name']} | Strength: {strength:.2f}")
            return final.tolist()

    raise Exception("No matching mixing rule found.")

# MIDI CC sending
def send_cc_gains(gains):
    port_name = config["MIDI_OUT_PORT"]
    midi_channel = config["MIDI_OUT_CHANNEL"]
    cc_dry = config["CC_DRY"]
    cc_fx = [config[f"CC_FX_{i+1}"] for i in range(config["NUM_FX_CHANNELS"])]

    num_fx = config["NUM_FX_CHANNELS"]
    dry_gain = gains[0]
    fx_gains = np.array(gains[1:], dtype=float)

    # Avoid division by zero
    if np.sum(fx_gains) == 0:
        fx_gains[:] = 1  # equal distribution

    # Normalize and rescale to sum to num_fx * 127
    fx_gains /= np.sum(fx_gains)
    fx_gains *= (num_fx * 127)

    # Clip and round
    fx_gains = np.clip(np.round(fx_gains), 0, 127).astype(int)

    with mido.open_output(port_name) as outport:
        outport.send(mido.Message('control_change', channel=midi_channel, control=cc_dry, value=int(np.clip(dry_gain, 0, 127))))
        for i, fx_cc in enumerate(cc_fx):
            outport.send(mido.Message('control_change', channel=midi_channel, control=fx_cc, value=fx_gains[i]))

# Inference thread
def inference_loop():
    while True:
        time.sleep(0.5)
        va = compute_va_per_channel(model, channel_buffers, device=device)
        gains = get_next_mix(va)
        send_cc_gains(gains)
        print("Gains (dry + FX + mix):", gains)

        for i, (v, a) in enumerate(va):
            cname = 'Dry' if i == 0 else 'Full Mix' if i == len(va) - 1 else f'FX {i}'
            print(f"{cname}: Valence = {v:.3f}, Arousal = {a:.3f}")
        print(f'Stress : {current_stress} Attention : {current_attention} Footpedal : {current_strength}')
        print('-----------')

# Zero gains initially
send_cc_gains([0] * NUM_CHANNELS)

# Start background inference
threading.Thread(target=inference_loop, daemon=True).start()

# Start stream and read MIDI
print("Running real-time inference. Press Ctrl+C to stop.")
with stream:
    while True:
        time.sleep(0.1)
        if inport is not None:
            for msg in inport.iter_pending():
                if msg.type == 'control_change':
                    if msg.control == config['CC_STRESS']:
                        current_stress = msg.value
                    elif msg.control == config['CC_ATTENTION']:
                        current_attention = msg.value
                    elif msg.control == config['FOOTPEDAL_CC']:
                        current_strength = msg.value