def main():
    config = load_config()
    files = get_files_to_process(config['input_folder'])
    
    print(f"Found {len(files)} files to process.")
    
    for file in files:
        try:
            process_file(file, config)
            print(f"Processed: {file}")
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
    
    print("Processing complete. Check the Transcripts and Summaries folders for results.")

if __name__ == "__main__":
    main()