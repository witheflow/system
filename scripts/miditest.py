import mido
import time
# Replace with the exact name of your IAC port from get_output_names()
port_name = 'IAC Driver Bus 1'

# Open the port
outport = mido.open_output(port_name)

# Send a Control Change message
cc_number = 21
cc_value = 0
channel = 0  # MIDI channel 1 is channel 0 in mido


while True:
    msg = mido.Message('control_change', control=cc_number, value=0, channel=channel)
    outport.send(msg)

    print(f"Sent CC {cc_number} with value {cc_value} on channel {channel + 1} to {port_name}")
    time.sleep(1)
    msg = mido.Message('control_change', control=cc_number, value=127, channel=channel)
    outport.send(msg)

    print(f"Sent CC {cc_number} with value 127 on channel {channel + 1} to {port_name}")
    time.sleep(1)
