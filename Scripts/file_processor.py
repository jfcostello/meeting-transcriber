import os
import shutil
from datetime import datetime
from .audio_extractor import extract_audio
from .transcriber import transcribe_audio
from .summarizer import summarize_transcript

def add_timestamp_to_filename(filename, config):
    if config.get('add_timestamp') == True:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-")
        if not filename.startswith(timestamp):  # Check if the exact timestamp is already present
            return f"{timestamp}{filename}"
    return filename

def process_videos(queue_folder, config):
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    for filename in os.listdir(queue_folder):
        if any(filename.lower().endswith(ext) for ext in video_extensions):
            new_filename = add_timestamp_to_filename(filename, config)
            old_path = os.path.join(queue_folder, filename)
            new_path = os.path.join(queue_folder, new_filename)
            os.rename(old_path, new_path)
            
            try:
                print(f"Processing video: {new_filename}")
                audio_path = extract_audio(new_path, queue_folder)
                move_file(new_path, config['processed_video_folder'])
                print(f"Video processed and moved: {new_filename}")
            except Exception as e:
                print(f"Error processing video {new_filename}: {str(e)}")

def process_audio_files(queue_folder, config):
    audio_extensions = ['.mp3', '.wav', '.m4a', '.flac']
    for filename in os.listdir(queue_folder):
        if any(filename.lower().endswith(ext) for ext in audio_extensions):
            new_filename = add_timestamp_to_filename(filename, config)
            old_path = os.path.join(queue_folder, filename)
            new_path = os.path.join(queue_folder, new_filename)
            os.rename(old_path, new_path)
            
            try:
                print(f"Processing audio: {new_filename}")
                transcript_path = transcribe_audio(new_path, queue_folder, config['whisper'])
                move_file(new_path, config['processed_audio_folder'])
                print(f"Audio processed and moved: {new_filename}")
            except Exception as e:
                print(f"Error processing audio {new_filename}: {str(e)}")

def process_transcripts(queue_folder, config):
    for filename in os.listdir(queue_folder):
        if filename.endswith('_transcript.md'):
            new_filename = add_timestamp_to_filename(filename, config)
            old_path = os.path.join(queue_folder, filename)
            new_path = os.path.join(queue_folder, new_filename)
            os.rename(old_path, new_path)
            
            try:
                print(f"Processing transcript: {new_filename}")
                summary_path = summarize_transcript(new_path, config)
                move_file(new_path, config['transcripts_folder'])
                print(f"Transcript processed and moved: {new_filename}")
            except Exception as e:
                print(f"Error processing transcript {new_filename}: {str(e)}")

def move_file(source_path, destination_folder):
    filename = os.path.basename(source_path)
    destination_path = os.path.join(destination_folder, filename)
    
    # If file with same name exists, append a number
    counter = 1
    while os.path.exists(destination_path):
        name, ext = os.path.splitext(filename)
        destination_path = os.path.join(destination_folder, f"{name}_{counter}{ext}")
        counter += 1

    shutil.move(source_path, destination_path)
    print(f"Moved file to: {destination_path}")