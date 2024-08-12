import os
import shutil
from audio_extractor import extract_audio
from transcriber import transcribe_audio
from summarizer import summarize_transcript
from config_handler import get_config

def process_file(file_path):
    config = get_config()
    file_name = os.path.basename(file_path)
    file_ext = os.path.splitext(file_name)[1].lower()

    print(f"Processing file: {file_name}")

    try:
        if file_ext in ['.mp4', '.avi', '.mov', '.mkv']:
            # Process video file
            audio_path = extract_audio(file_path, config['processed_audio_folder'])
            transcript_path = transcribe_audio(audio_path, config['transcripts_folder'], config['whisper_model'], config['language'])
            summary_path = summarize_transcript(transcript_path, config)
            move_processed_file(file_path, config['processed_video_folder'])
            print(f"Video processed: {file_name}")
            print(f"Audio extracted: {os.path.basename(audio_path)}")
            print(f"Transcript created: {os.path.basename(transcript_path)}")
            print(f"Summary created: {os.path.basename(summary_path)}")
        elif file_ext in ['.mp3', '.wav', '.m4a', '.flac']:
            # Process audio file
            transcript_path = transcribe_audio(file_path, config['transcripts_folder'], config['whisper_model'], config['language'])
            summary_path = summarize_transcript(transcript_path, config)
            move_processed_file(file_path, config['processed_audio_folder'])
            print(f"Audio processed: {file_name}")
            print(f"Transcript created: {os.path.basename(transcript_path)}")
            print(f"Summary created: {os.path.basename(summary_path)}")
        else:
            print(f"Unsupported file type: {file_ext}")
    except Exception as e:
        print(f"Error processing {file_name}: {str(e)}")

def move_processed_file(file_path, destination_folder):
    file_name = os.path.basename(file_path)
    destination_path = os.path.join(destination_folder, file_name)
    
    # If file with same name exists, append a number
    counter = 1
    while os.path.exists(destination_path):
        name, ext = os.path.splitext(file_name)
        destination_path = os.path.join(destination_folder, f"{name}_{counter}{ext}")
        counter += 1

    shutil.move(file_path, destination_path)
    print(f"Moved processed file to: {destination_path}")

def get_files_to_process(folder_path):
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
            if os.path.isfile(os.path.join(folder_path, f))]