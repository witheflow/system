import mido
import time

# Replace with your REAPER virtual MIDI input name
for n in mido.get_input_names():
    print(n)

