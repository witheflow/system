import argparse
import os

import pandas as pd
import re


def parse_session_log(log_file_path: str) -> pd.DataFrame:
    """
    Parse session_log.txt and convert to a pandas DataFrame.
    Dynamically handles varying channel counts and names.
    
    Args:
        log_file_path (str): Path to the session_log.txt file
        
    Returns:
        pd.DataFrame: DataFrame containing parsed data with one row per packet
    """
    # Read the file
    with open(log_file_path, 'r') as file:
        lines = file.readlines()
    
    # Find the start of the first data block (first "Active Rule" line)
    start_idx = 0
    for i, line in enumerate(lines):
        if "Active Rule:" in line:
            start_idx = i
            break
    
    # If no data blocks found, return empty DataFrame
    if start_idx == len(lines):
        return pd.DataFrame()
    
    # Parse the first block to determine the structure
    block_info = find_block_boundaries(lines, start_idx)
    if not block_info:
        return pd.DataFrame()
    
    block_start, block_end, channel_names, num_channels = block_info
    
    # Initialize data storage
    data = []
    
    # Process all blocks
    i = start_idx
    while i < len(lines):
        if "Active Rule:" not in lines[i]:
            i += 1
            continue
        
        block_info = find_block_boundaries(lines, i)
        if not block_info:
            break
        
        block_start, block_end, current_channel_names, current_num_channels = block_info
        
        # Validate block structure
        expected_lines = num_channels + 5
        actual_lines = block_end - block_start + 1
        
        if actual_lines != expected_lines:
            print(f"Warning: Block at line {i} has {actual_lines} lines, expected {expected_lines}")
            i = block_end + 1
            continue
            
        # Validate channel names consistency
        if current_channel_names != channel_names:
            print(f"Warning: Channel names changed at line {i}")
            print(f"Original: {channel_names}")
            print(f"Current: {current_channel_names}")
            i = block_end + 1
            continue
        
        # Parse this block
        block_data = parse_block(lines[block_start:block_end+1], channel_names)
        if block_data:
            data.append(block_data)
        
        i = block_end + 1
    
    # Convert to DataFrame
    if not data:
        return pd.DataFrame()
    
    # Create base DataFrame
    df = pd.DataFrame(data)
    
    return df

def find_block_boundaries(lines: list[str], start_idx: int) -> (int, int, list[str], int):
    """
    Find the boundaries of a data block and extract channel names.
    
    Args:
        lines: All lines from the log file
        start_idx: Index of the "Active Rule" line
    
    Returns:
        Tuple containing:
        - block_start: Index of the first line in the block
        - block_end: Index of the last line in the block
        - channel_names: List of channel names
        - num_channels: Number of channels
    """
    block_start = start_idx
    
    # First line should have "Active Rule"
    if block_start >= len(lines) or "Active Rule:" not in lines[block_start]:
        return None
    
    # Third line should have gains list
    gains_idx = block_start + 2
    if gains_idx >= len(lines):
        return None
    
    # Extract gains to determine channel count
    gains_match = re.search(r'\[([\d, ]+)\]', lines[gains_idx])
    if not gains_match:
        return None
    
    gains_str = gains_match.group(1)
    gains = [int(g.strip()) for g in gains_str.split(',')]
    num_channels = len(gains)
    
    # Extract channel names from the following lines
    channel_names = []
    for i in range(num_channels):
        idx = gains_idx + 1 + i
        if idx >= len(lines):
            return None
        
        line = lines[idx]
        channel_match = re.search(r'^\[\d+:\d+:\d+\.\d+[^\w]*([^:]+):', line)
        if channel_match:
            channel_name = channel_match.group(1).strip()
            channel_names.append(channel_name)
    
    # Check if we got the expected number of channel names
    if len(channel_names) != num_channels:
        return None
    
    # Last line should be dashes
    block_end = gains_idx + num_channels + 2
    if block_end >= len(lines) or "---" not in lines[block_end]:
        return None
    
    return block_start, block_end, channel_names, num_channels

def parse_block(block_lines: list[str], channel_names: list[str]) -> dict:
    """
    Parse a single data block into a dictionary.

    Args:
        block_lines: Lines from a single data block
        channel_names: List of channel names

    Returns:
        Dictionary containing parsed data from the block
    """
    # Initialize result dictionary
    result = {}

    # Extract timestamp
    timestamp_match = re.match(r'\[(\d+:\d+:\d+\.\d+)', block_lines[0])
    if timestamp_match:
        result['timestamp'] = timestamp_match.group(1)
    else:
        result['timestamp'] = None

    # Extract rule and strength
    rule_match = re.search(r'Active Rule: (.+) \| Strength: ([\d\.]+)', block_lines[0])
    if rule_match:
        result['rule'] = rule_match.group(1)
        result['strength'] = float(rule_match.group(2))
    else:
        result['rule'] = None
        result['strength'] = None

    # Extract gains
    gains_match = re.search(r'\[([\d, ]+)\]', block_lines[2])
    if gains_match:
        gains_str = gains_match.group(1)
        gains = [int(g.strip()) for g in gains_str.split(',')]

        # Add gains to result, using channel names
        for i, channel in enumerate(channel_names):
            if i < len(gains):
                result[f'gain_{channel.lower().replace(" ", "_")}'] = gains[i]

    # Extract valence and arousal values
    num_channels = len(channel_names)

    # Initialize all valence/arousal values to NaN
    for channel in channel_names:
        channel_key = channel.lower().replace(" ", "_")
        result[f'{channel_key}_valence'] = float('nan')
        result[f'{channel_key}_arousal'] = float('nan')

    # Keep track of matched channels
    matched_channels = set()

    # Process the valence/arousal lines
    for i in range(num_channels):
        line_idx = i + 3
        if line_idx >= len(block_lines):
            break

        line = block_lines[line_idx]
        # Extract both channel name and VA values
        full_match = re.search(r'\[\d+:\d+:\d+\.\d+.*?([^:]+): Valence = ([-\d\.]+), Arousal = ([-\d\.]+)', line)
        if not full_match:
            print(f"Warning: Failed to parse line: {line}")

        if full_match:
            detected_channel = full_match.group(1).strip()
            valence = float(full_match.group(2))
            arousal = float(full_match.group(3))

            # Check if this channel name is in our expected channel names
            if detected_channel in channel_names:
                channel_key = detected_channel.lower().replace(" ", "_")

                # Check for duplicates
                if detected_channel in matched_channels:
                    print(f"Warning: Duplicate channel '{detected_channel}' found in block")

                matched_channels.add(detected_channel)
                result[f'{channel_key}_valence'] = valence
                result[f'{channel_key}_arousal'] = arousal
            else:
                print(f"Warning: Unexpected channel '{detected_channel}' found in block")

    # Check for unmatched channels
    unmatched = set(channel_names) - matched_channels
    if unmatched:
        print(f"Warning: Channels not found in block: {', '.join(unmatched)}")

    # Extract stress, attention, and footpedal values
    sensor_line_idx = 3 + num_channels
    if sensor_line_idx < len(block_lines):
        sensor_match = re.search(
            r'Stress : (\d+) Attention : (\d+) Footpedal : (\d+)',
            block_lines[sensor_line_idx]
        )
        if sensor_match:
            result['stress'] = int(sensor_match.group(1))
            result['attention'] = int(sensor_match.group(2))
            result['footpedal'] = int(sensor_match.group(3))

    return result

if __name__ == "__main__":
    # Create command line argument parser
    parser = argparse.ArgumentParser(description='Parse session log file into a pandas DataFrame')
    parser.add_argument('-i', '--input', type=str,
                        default='/home/jason/projects/research/witheflow/scripts/session_log.txt',
                        help='Path to session_log.txt file')
    parser.add_argument('-o', '--output', type=str,
                        default='/home/jason/projects/research/witheflow/scripts/session_data.csv',
                        help='Path to output CSV file')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose output (detailed parsing information)')
    args = parser.parse_args()

    # Parse the session log and create a DataFrame
    log_path = args.input
    if not os.path.exists(log_path):
        print(f"Error: Input file '{log_path}' not found.")
        exit(1)

    df = parse_session_log(log_path)

    # Display information about the DataFrame
    if not df.empty:
        print(f"Parsed {len(df)} data blocks")

        # Display channel names
        channel_names = []
        for col in df.columns:
            if col.startswith('gain_'):
                channel_name = col[5:].replace('_', ' ')
                channel_names.append(channel_name)

        print(f"Found {len(channel_names)} channels: {', '.join(channel_names)}")
        print(f"Data columns: {len(df.columns)}")

        print("\nFirst 5 rows:")
        print(df.head())

        # Save the DataFrame to a CSV file
        csv_path = args.output
        df.to_csv(csv_path, index=False)
        print(f"\nSaved DataFrame to {csv_path} with {len(df)} rows")
    else:
        print("No data blocks found or parsing failed.")