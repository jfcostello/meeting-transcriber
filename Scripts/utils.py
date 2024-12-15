import os
import shutil
from datetime import datetime
from .config_handler import get_config

def move_file(source_path, config):
    filename = os.path.basename(source_path)
    
    # Load output structure config
    output_config = config.get('output_structure', {})
    base_folder = output_config.get('base_folder', 'output')
    structure = output_config.get('structure', [])
    
    # Extract base filename without extension or suffixes
    base_filename = filename
    suffixes = ["_transcript_summary", "_transcript", "_summary"]
    for suffix in suffixes:
        if base_filename.endswith(suffix + ".md"):
            base_filename = base_filename[:-len(suffix + ".md")]
    base_filename = os.path.splitext(base_filename)[0]
    
    # Extract parent directory name
    parent_dir = os.path.basename(os.path.dirname(source_path))
    
    # Get today's date in YYYY-MM-DD format
    today_date = datetime.now().strftime("%Y-%m-%d")
    
    # Construct the output directory structure
    output_path_parts = [base_folder]
    for item in structure:
        if item == "DATE":
            output_path_parts.append(today_date)
        elif item == "FILE-NAME":
             output_path_parts.append(base_filename)
        elif item == "SUMMARY-TYPE":
            output_path_parts.append(parent_dir)
    
    output_dir = os.path.join(*output_path_parts)
    os.makedirs(output_dir, exist_ok=True)
    
    destination_path = os.path.join(output_dir, filename)

    shutil.move(source_path, destination_path)
    print(f"Moved file to: {destination_path}")
