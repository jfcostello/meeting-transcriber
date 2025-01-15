import os
import shutil
from datetime import datetime
from .audio_extractor import extract_audio
from .transcriber import transcribe_audio_flow
from .summarizer import summarize_transcript
from .config_handler import get_config
from .utils import move_file

def add_timestamp_to_filename(filename, config):
    if config.get('add_timestamp') == True:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-")
        if not filename.startswith(timestamp):  # Check if the exact timestamp is already present
            return f"{timestamp}{filename}"
    return filename

def process_videos(queue_folder, config):
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    
    # Walk through all directories recursively
    for root, dirs, files in os.walk(queue_folder):
        # Check if current directory has summary-rules.txt
        summary_rules_path = os.path.join(root, "summary-rules.txt")
        if os.path.exists(summary_rules_path):
            # Process video files in this directory
            for filename in files:
                if any(filename.lower().endswith(ext) for ext in video_extensions):
                    new_filename = add_timestamp_to_filename(filename, config)
                    old_path = os.path.join(root, filename)
                    new_path = os.path.join(root, new_filename)
                    os.rename(old_path, new_path)
                    
                    try:
                        print(f"Processing video: {new_filename}")
                        audio_path = extract_audio(new_path, queue_folder)
                        move_file(new_path, config)
                        print(f"Video processed and moved: {new_filename}")
                    except Exception as e:
                        print(f"Error processing video {new_filename}: {str(e)}")
        else:
            # Skip directories without summary-rules.txt
            continue

def process_audio_files(queue_folder, config):
    audio_extensions = ['.mp3', '.wav', '.m4a', '.flac']
    
    # Walk through all directories recursively
    for root, dirs, files in os.walk(queue_folder):
        # Check if current directory has summary-rules.txt
        summary_rules_path = os.path.join(root, "summary-rules.txt")
        if os.path.exists(summary_rules_path):
            # Process audio files in this directory
            for filename in files:
                if any(filename.lower().endswith(ext) for ext in audio_extensions):
                    new_filename = add_timestamp_to_filename(filename, config)
                    old_path = os.path.join(root, filename)
                    new_path = os.path.join(root, new_filename)
                    os.rename(old_path, new_path)
                    
                    try:
                        print(f"Processing audio: {new_filename}")
                        
                        transcript_path = transcribe_audio_flow(new_path, queue_folder, config)
                        move_file(new_path, config)
                        print(f"Audio processed and moved: {new_filename}")
                    except Exception as e:
                        print(f"Error processing audio {new_filename}: {str(e)}")
        else:
            # Skip directories without summary-rules.txt
            continue
                
def process_transcripts(queue_folder, config):
    # Walk through all directories recursively
    for root, dirs, files in os.walk(queue_folder):
        # Check if current directory has summary-rules.txt
        summary_rules_path = os.path.join(root, "summary-rules.txt")
        if os.path.exists(summary_rules_path):
            # Process transcript files in this directory
            for filename in files:
                if filename.endswith('_transcript.md'):
                    new_filename = add_timestamp_to_filename(filename, config)
                    old_path = os.path.join(root, filename)
                    new_path = os.path.join(root, new_filename)
                    if old_path != new_path:
                        os.rename(old_path, new_path)
                    
                    try:
                        print(f"Processing transcript: {new_filename}")
                        summary_path = summarize_transcript(new_path, config)
                        move_file(new_path, config)
                        print(f"Transcript processed and moved: {new_filename}")
                    except Exception as e:
                        print(f"Error processing transcript {new_filename}: {str(e)}")
        else:
            # Skip directories without summary-rules.txt
            continue
