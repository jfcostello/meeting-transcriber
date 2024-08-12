import os
from scripts.file_processor import process_videos, process_audio_files, process_transcripts
from scripts.config_handler import get_config

def main():
    config = get_config()
    queue_folder = config['meeting_recordings_folder']

    print("Starting processing pipeline...")

    # Process videos
    print("\nProcessing videos...")
    process_videos(queue_folder, config)

    # Process audio files
    print("\nProcessing audio files...")
    process_audio_files(queue_folder, config)

    # Process transcripts
    print("\nProcessing transcripts...")
    process_transcripts(queue_folder, config)

    print("\nProcessing complete. Check the respective folders for results.")

if __name__ == "__main__":
    main()