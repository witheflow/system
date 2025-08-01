# witheflow (mac only)

## Python Setup

1. Install Python (e.g. with [pyenv](https://github.com/pyenv/pyenv)).
2. Create the virtualenv:

   ```bash
   pyenv virtualenv 3.10.9 witheflow
   pyenv activate witheflow
   pip install -r requirements.txt
   ```
3. Download the [model](https://www.kaggle.com/models/google/yamnet/tfLite/classification-tflite/1?tfhub-redirect=true). 
4. Set the environemnt variable ```WITHEFLOW_MODEL``` to the path where you saved the model. e.g. 
```
export WITHEFLOW_MODEL=PANN
```

5. Set the environment variable ```WITHEFLOW_DEVICE_INDEX``` to the blackhole audio device you create in the next step (to list available devices run

```
import sounddevice as sd

# List all available audio devices
print(sd.query_devices())
```
and then (assuming blackhole is device 0)

```
export WITHEFLOW_DEVICE_INDEX = 0 
```


 ## Audio Setup (mac)
 1. Install [blackhole](https://existential.audio/blackhole/).
 2. Open **Audio MIDI setup** (mac settings)
 3. Create new **Aggregate device**
 4. Tick `BlackHole 16ch` and the default device (e.g. Macbook Air Speakers, or the audio interface)
 5. Make note of the channels (e.g. for me Macbook Air speakers are in outputs 1,2, and BlackHole is 3-18)

 ## Reaper Setup
 0. Set the audio device to the aggregate device we just created
 1. Create 3 tracks. The first will be the musician's audio. Set the input accordingly. Don't touch the output.
 2. The other two will be set with different FX (do whatever you want).
 3. Route the first track to each of the other two.

 After this setup, if you run ```python audioset_classfier.py``` in the ```scripts``` folder, you will be shown the classification for each of the two FX tracks in real time (you need to send audio through the first track)

 ## TODO
 [] MIDI control of reaper faders from python

 [] Swap classifier for something more meaningful
 
 [] Implement the logic for attenuating the faders
 
