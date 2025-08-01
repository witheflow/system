import sounddevice as sd
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import audio
from mediapipe.tasks.python.audio import AudioClassifier, AudioClassifierOptions, RunningMode
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.components.containers import audio_data
import os
import mido



devices = sd.query_devices()
for i, device in enumerate(devices):
    print(f"{i}: {device['name']} ({device['max_input_channels']} in, {device['max_output_channels']} out)")
# Config
port_name = 'IAC Driver Bus 1'
# Open the port
outport = mido.open_output(port_name)
cc_number_FX1 = 20
cc_number_FX2 = 21
channel = 0  # MIDI channel 1 is channel 0 in mido
SAMPLE_RATE = 16000
CHANNELS = 2
BLOCK_DURATION = 1.0
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION)
# Path to the model

# List all available audio devices
model_path = os.getenv('WITHEFLOW_MODEL','../model/1.tflite')
DEVICE_INDEX = os.getenv('WITHEFLOW_DEVICE_INDEX', 0)

# Shared results
results = {"ch1": None, "ch2": None}

# 🎵 Music-related labels
music_labels = {
    'jazz', 'rock music', 'hip hop music', 'classical music', 'country music',
    'pop music', 'electronic music', 'heavy metal', 'reggae', 'disco',
    'blues', 'funk', 'techno', 'punk rock', 'folk music', 'soul music',
    'gospel music', 'latin music', 'ska', 'house music', 'trance music',
    'ambient music', 'grunge', 'new wave',
    # Instruments
    'guitar', 'piano', 'violin', 'drums', 'saxophone', 'trumpet', 'cello', 'flute',
    'harmonica', 'banjo', 'accordion', 'harp', 'synthesizer',
    # Musical acts
    'singing', 'rapping', 'beatboxing', 'humming', 'whistling', 'clapping'
}
# Add this near the top
target_label = 'guitar'  # 🎯 Set your desired label here

# Update the callback inside make_result_callback
def make_result_callback(channel_key):
    def callback(result: audio.AudioClassifierResult, timestamp_ms: int):
        if result.classifications:
            for cat in result.classifications[0].categories:
                if cat.category_name.lower() == target_label.lower():
                    results[channel_key] = (cat.category_name, cat.score)
                    break
            else:
                results[channel_key] = (target_label, 0.0)
    return callback

# Update the display section in the main callback
def callback(indata, frames, time_info, status):
    if status:
        print(f"⚠️ {status}")

    ch1 = indata[:, 0].astype(np.float32)
    ch2 = indata[:, 1].astype(np.float32)

    ch1 /= np.max(np.abs(ch1)) + 1e-8
    ch2 /= np.max(np.abs(ch2)) + 1e-8

    timestamp_ms = int(time_info.inputBufferAdcTime * 1000)
    classifier1.classify_async(audio_data.AudioData.create_from_array(ch1, SAMPLE_RATE), timestamp_ms)
    classifier2.classify_async(audio_data.AudioData.create_from_array(ch2, SAMPLE_RATE), timestamp_ms)

    res1 = results["ch1"] if results["ch1"] else (target_label, 0.0)
    res2 = results["ch2"] if results["ch2"] else (target_label, 0.0)
    if res1>res2:
        outport.send(mido.Message('control_change', control=cc_number_FX1, value=127, channel=channel))
        outport.send(mido.Message('control_change', control=cc_number_FX2, value=0, channel=channel))
    else:
        outport.send(mido.Message('control_change', control=cc_number_FX1, value=0, channel=channel))
        outport.send(mido.Message('control_change', control=cc_number_FX2, value=127, channel=channel))
    print(f"🔊 ch1: {res1[0]} ({res1[1]:.5f}) | ch2: {res2[0]} ({res2[1]:.5f})")


def create_classifier(channel_key):
    opts = AudioClassifierOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.AUDIO_STREAM,
        result_callback=make_result_callback(channel_key),
        max_results=5  # get more to allow filtering
    )
    return AudioClassifier.create_from_options(opts)

# Create two classifiers
classifier1 = create_classifier("ch1")
classifier2 = create_classifier("ch2")

# === Audio Callback ===


# === Start Listening ===

with sd.InputStream(
    channels=CHANNELS,
    samplerate=SAMPLE_RATE,
    blocksize=BLOCK_SIZE,
    callback=callback,
    device=DEVICE_INDEX
):
    print("🎧 Classifying each channel independently (top 2 music-related labels only)...")
    while True:
        sd.sleep(50)
