import os
import json
import time
import sys
from datetime import datetime
import threading
from collections import deque
from queue import Queue, Empty

import numpy as np
import sounddevice as sd
import torch
import yaml
import mido

# pyenv activate witheflow

# ----------------------------
# Globals set in main()
# ----------------------------
config = None
SAMPLE_RATE = 32000
FRAME_DURATION = 0.25
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION)
BUFFER_DURATION = 5.0
NUM_CHANNELS = None

previous_gains = None
current_stress = 0
current_attention = 0
current_strength = 127
previous_dry_va = None
dry_va_history = deque(maxlen=20)
stable_seconds = 0

channel_buffers = None
stream = None
model = None
device = "cpu"
inport = None
OUTPORT = None

# path for dashboard JSON (shared between processes)
STATE_JSON_PATH = os.path.abspath("viz_latest.json")


# ----------------------------
# Logger
# ----------------------------
class DualLogger:
    def __init__(self, logfile_path):
        self.terminal = sys.__stdout__
        self.logfile = open(logfile_path, "a", buffering=1)  # line-buffered

    def write(self, message):
        self.terminal.write(message)
        for line in message.rstrip().splitlines():
            if line.strip():
                ts = datetime.now().strftime("[%H:%M:%S.%f]")[:-3]
                self.logfile.write(f"{ts} {line}\n")

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()


# ----------------------------
# Safe JSON write for dashboard
# ----------------------------
def write_state_json(path, payload):
    # atomic write: write to tmp then rename
    tmp = f"{path}.tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, path)
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


# ----------------------------
# Audio buffer
# ----------------------------
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


def audio_callback(indata, frames, time_info, status):
    for i, buf in enumerate(channel_buffers):
        buf.update(indata[:, i])


# ----------------------------
# Model / VA
# ----------------------------
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


# ----------------------------
# Gain functions
# ----------------------------
def boost_far_in_high_arousal_direction(dry_va, fx_vas, **kwargs):
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


# ----------------------------
# Rules
# ----------------------------
def matches_condition(rule_cond, x_dict):
    for key, cond in rule_cond.items():
        val = x_dict[key]
        if "gt" in cond and not val > cond["gt"]:
            return False
        if "lt" in cond and not val < cond["lt"]:
            return False
    return True


def get_next_mix(valence_arousal):
    global previous_dry_va, stable_seconds, previous_gains, current_strength, ACTIVE_RULE_NAME

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

            strength = current_strength / 127.0
            inverted = 127 - smoothed
            final = np.round(strength * smoothed + (1 - strength) * inverted).astype(int)

            previous_gains[:] = final
            ACTIVE_RULE_NAME = rule['name']
            # print(f"Active Rule: {rule['name']} | Strength: {strength:.2f}")
            return final.tolist()

    raise Exception("No matching mixing rule found.")


# ----------------------------
# MIDI
# ----------------------------
def send_cc_gains(gains):
    """Uses global OUTPORT opened once in main()."""
    midi_channel = config["MIDI_OUT_CHANNEL"]
    cc_dry = config["CC_DRY"]
    cc_fx = [config[f"CC_FX_{i+1}"] for i in range(config["NUM_FX_CHANNELS"])]

    num_fx = config["NUM_FX_CHANNELS"]
    dry_gain = gains[0]
    fx_gains = np.array(gains[1:], dtype=float)

    if np.sum(fx_gains) == 0:
        fx_gains[:] = 1

    fx_gains /= np.sum(fx_gains)
    fx_gains *= (num_fx * 127)
    fx_gains = np.clip(np.round(fx_gains), 0, 127).astype(int)

    OUTPORT.send(mido.Message('control_change', channel=midi_channel, control=cc_dry, value=int(np.clip(dry_gain, 0, 127))))
    for i, fx_cc in enumerate(cc_fx):
        OUTPORT.send(mido.Message('control_change', channel=midi_channel, control=fx_cc, value=int(fx_gains[i])))


# ----------------------------
# Inference loop
# ----------------------------
ACTIVE_RULE_NAME = ""

def inference_loop():
    while True:
        time.sleep(0.5)
        va = compute_va_per_channel(model, channel_buffers, device=device)
        gains = get_next_mix(va)
        send_cc_gains(gains)

        state = {
            "va": va.tolist(),
            "gains": [int(x) for x in gains],
            "stress": int(current_stress),
            "attention": int(current_attention),
            "strength": int(current_strength),
            "rule": ACTIVE_RULE_NAME,
        }
        write_state_json(STATE_JSON_PATH, state)


# ----------------------------
# Browser dashboard (Dash)
# ----------------------------
def run_dashboard(json_path, num_fx_channels, host="127.0.0.1", port=8051):
    from dash import Dash, dcc, html, Output, Input
    import plotly.graph_objects as go
    import json

    labels = ["Dry"] + [f"FX{i}" for i in range(1, num_fx_channels+1)] + ["Mix"]

    def base_va_fig():
        fig = go.Figure()
        fig.update_layout(
            margin=dict(l=40, r=10, t=40, b=40),
            xaxis=dict(range=[-1, 1], title="Valence", zeroline=True, zerolinewidth=1),
            yaxis=dict(range=[-1, 1], title="Arousal", zeroline=True, zerolinewidth=1),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=450,
        )
        # Fixed trace order: Dry, FX..., Full
        fig.add_trace(go.Scatter(x=[], y=[], mode="markers", name="Dry", marker=dict(size=10)))
        for i in range(1, num_fx_channels+1):
            fig.add_trace(go.Scatter(x=[], y=[], mode="markers", name=f"FX {i}", marker=dict(size=9, symbol="triangle-up")))
        fig.add_trace(go.Scatter(x=[], y=[], mode="markers", name="Full Mix", marker=dict(size=10, symbol="square")))
        return fig

    def base_gains_fig():
        fig = go.Figure(go.Bar(x=labels, y=[0]*(num_fx_channels+2)))
        fig.update_layout(
            margin=dict(l=40, r=10, t=40, b=40),
            yaxis=dict(range=[0, 127], title="Gain (CC)"),
            height=450,
        )
        return fig

    app = Dash(__name__)
    app.title = "witheFlow HUD"

    app.layout = html.Div([
        html.H3("witheFlow Real-time HUD"),
        html.Div([
            dcc.Graph(id="va-graph", figure=base_va_fig(), style={"width": "60%", "display": "inline-block"}),
            dcc.Graph(id="gains-graph", figure=base_gains_fig(), style={"width": "39%", "display": "inline-block"}),
        ]),
        html.Div(id="hud-text", style={"fontFamily": "monospace", "marginTop": "8px"}),
        dcc.Interval(id="tick", interval=500, n_intervals=0),  # 2 Hz
    ], style={"padding": "10px"})

    @app.callback(
        Output("va-graph", "figure"),
        Output("gains-graph", "figure"),
        Output("hud-text", "children"),
        Input("tick", "n_intervals"),
        prevent_initial_call=False
    )
    def _update_figures(_n):
        # Build fresh figures every tick (avoids Plotly "data" mutation issues)
        va_fig = base_va_fig()
        gains_fig = base_gains_fig()

        try:
            with open(json_path, "r") as f:
                state = json.load(f)
        except Exception:
            return va_fig, gains_fig, "Waiting for data…"

        va = state.get("va", [])
        gains = state.get("gains", [])
        stress = state.get("stress", 0)
        attention = state.get("attention", 0)
        strength = state.get("strength", 127)
        rule = state.get("rule", "")

        # Update VA points
        if isinstance(va, list) and len(va) >= 2 and isinstance(va[0], list):
            try:
                dry = va[0]
                full = va[-1]
                fx_list = va[1:-1]
                va_fig.data[0].x = [dry[0]];  va_fig.data[0].y = [dry[1]]
                for i in range(min(len(fx_list), num_fx_channels)):
                    va_fig.data[1+i].x = [fx_list[i][0]]
                    va_fig.data[1+i].y = [fx_list[i][1]]
                va_fig.data[1+num_fx_channels].x = [full[0]]
                va_fig.data[1+num_fx_channels].y = [full[1]]
            except Exception:
                pass

        # Update gains bar
        expected = num_fx_channels + 2
        if not isinstance(gains, list):
            gains = [0]*expected
        elif len(gains) < expected:
            gains = gains + [0]*(expected - len(gains))
        elif len(gains) > expected:
            gains = gains[:expected]
        gains_fig = go.Figure(go.Bar(x=labels, y=gains))
        gains_fig.update_layout(
            margin=dict(l=40, r=10, t=40, b=40),
            yaxis=dict(range=[0, 127], title="Gain (CC)"),
            height=450,
        )

        hud = f"Rule: {rule or '-'}   |   Stress: {int(stress):3d}   Attention: {int(attention):3d}   Footpedal: {int(strength):3d}"
        return va_fig, gains_fig, hud

    # Dash ≥2.12
    try:
        app.run(host=host, port=port, debug=False)
    except TypeError:
        app.run_server(host=host, port=port, debug=False)







# ----------------------------
# Main (spawn-safe)
# ----------------------------
def main():
    global config, NUM_CHANNELS, previous_gains, channel_buffers, stream
    global model, device, inport, OUTPORT

    # logger
    sys.stdout = DualLogger("session_log.txt")

    # load config
    with open('config.json', 'r') as f:
        config = json.load(f)

    NUM_CHANNELS = config["NUM_FX_CHANNELS"] + 2
    previous_gains = [0.0] * NUM_CHANNELS

    # list devices (once)
    print("Available MIDI output ports:")
    for name in mido.get_output_names():
        print(name)
    for idx, dev in enumerate(sd.query_devices()):
        print(f"{idx}: {dev['name']} ({dev['max_input_channels']} in, {dev['max_output_channels']} out)")

    # rules
    if config['SENSORS_AVAILABLE'] in ("all", "attention"):
        inport = mido.open_input(config['MIDI_IN_PORT'])
        rules_file = "rules_stress_attention.yaml" if config['SENSORS_AVAILABLE'] == "all" else "rules_attention.yaml"
    else:
        inport = None
        rules_file = "rules_audio.yaml"

    with open(rules_file, "r") as f:
        global rules
        rules = yaml.safe_load(f)["rules"]

    # buffers/stream
    global SAMPLE_RATE, FRAME_DURATION, FRAME_SIZE, BUFFER_DURATION
    channel_buffers = [AudioBuffer(BUFFER_DURATION, SAMPLE_RATE) for _ in range(NUM_CHANNELS)]
    stream = sd.InputStream(
        device=config["DEVICE_INDEX"],
        channels=NUM_CHANNELS,
        samplerate=SAMPLE_RATE,
        blocksize=FRAME_SIZE,
        callback=audio_callback
    )

    # model
    model_path = config["MODEL_PATH"]
    model = torch.jit.load(model_path, map_location="cpu")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)

    # MIDI out (open once)
    OUTPORT = mido.open_output(config["MIDI_OUT_PORT"])

    # --- start DASH dashboard in a separate process (macOS-safe) ---
    from multiprocessing import get_context
    ctx = get_context("spawn")
    dash_proc = ctx.Process(
        target=run_dashboard,
        args=(STATE_JSON_PATH, config["NUM_FX_CHANNELS"], "127.0.0.1",8051),
        daemon=True
    )
    dash_proc.start()
    time.sleep(0.5)

    # prime a dummy frame so you immediately see something
    write_state_json(STATE_JSON_PATH, {
        "va": [[0.0, 0.0]] + [[0.5, 0.5]] * config["NUM_FX_CHANNELS"] + [[-0.5, -0.5]],
        "gains": [0] + [64]*config["NUM_FX_CHANNELS"] + [0],
        "stress": 10, "attention": 20, "strength": 127,
        "rule": "boot"
    })

    # init zero gains
    send_cc_gains([0] * NUM_CHANNELS)

    # start inference
    threading.Thread(target=inference_loop, daemon=True).start()

    # main loop
    print("Running real-time inference. Open http://127.0.0.1:8051 for the HUD. Press Ctrl+C to stop.")
    try:
        with stream:
            while True:
                time.sleep(0.1)
                if inport is not None:
                    for msg in inport.iter_pending():
                        if msg.type == 'control_change':
                            if msg.control == config['CC_STRESS']:
                                globals()['current_stress'] = msg.value
                            elif msg.control == config['CC_ATTENTION']:
                                globals()['current_attention'] = msg.value
                            elif msg.control == config['FOOTPEDAL_CC']:
                                globals()['current_strength'] = msg.value
    finally:
        try:
            OUTPORT.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()


