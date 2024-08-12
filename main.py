from scripts.config_handler import load_config
from scripts.file_processor import get_files_to_process, process_file

def main():
    config = load_config()
    files = get_files_to_process(config['meeting_recordings_folder'])
    
    print(f"Found {len(files)} files to process.")
    
    for file in files:
        try:
            process_file(file)
            print(f"Processed: {file}")
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
    
    print("Processing complete. Check the Transcripts and Summaries folders for results.")

if __name__ == "__main__":
    main()