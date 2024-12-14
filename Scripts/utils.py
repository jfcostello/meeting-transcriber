import os
import shutil
from datetime import datetime

def move_file(source_path, config):
    filename = os.path.basename(source_path)
    output_folder = config.get('output_folder', '/output/')
    
    # Get today's date in YYYY-MM-DD format
    today_date = datetime.now().strftime("%Y-%m-%d")
    
    # Extract base filename without extension or suffixes
    base_filename = filename
    suffixes = ["_transcript_summary", "_transcript", "_summary"]
    for suffix in suffixes:
        if base_filename.endswith(suffix + ".md"):
            base_filename = base_filename[:-len(suffix + ".md")]
    base_filename = os.path.splitext(base_filename)[0]
    
    # Create the output directory structure
    date_folder = os.path.join(output_folder, today_date)
    file_folder = os.path.join(date_folder, base_filename)
    os.makedirs(file_folder, exist_ok=True)
    
    destination_path = os.path.join(file_folder, filename)

    shutil.move(source_path, destination_path)
    print(f"Moved file to: {destination_path}")
