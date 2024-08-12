import os
import sys
import traceback
import logging

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.config_handler import get_summary_prompt, get_config
from scripts.llm_utils import call_llm_api

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_unique_filename(base_path):
    directory, filename = os.path.split(base_path)
    name, ext = os.path.splitext(filename)
    counter = 1
    new_path = base_path
    while os.path.exists(new_path):
        new_path = os.path.join(directory, f"{name}_{counter}{ext}")
        counter += 1
    return new_path

def summarize_transcript(transcript_path, config):
    try:
        logger.info(f"Starting summarization for: {transcript_path}")
        logger.debug(f"Config: {config}")

        file_name = os.path.splitext(os.path.basename(transcript_path))[0]
        output_folder = config.get('summaries_folder')
        
        if not output_folder:
            raise ValueError("'summaries_folder' not found in config")
        
        base_output_path = os.path.join(output_folder, f"{file_name}_summary.md")
        output_path = get_unique_filename(base_output_path)

        # Ensure output folder exists
        os.makedirs(output_folder, exist_ok=True)

        # Read transcript
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript = f.read()

        # Get LLM configuration and summary prompt
        llm_config = config.get('llm')
        if not llm_config:
            raise ValueError("'llm' configuration not found in config")

        try:
            summary_prompt = get_summary_prompt(config)
        except Exception as e:
            logger.error(f"Error getting summary prompt: {str(e)}")
            raise

        # Call LLM API
        logger.debug(f"LLM Config: {llm_config}")
        logger.debug(f"Base URL from config: {llm_config.get('base_url')}")

        summary = call_llm_api(
            model=llm_config.get('model'),
            content=transcript,
            systemPrompt=summary_prompt,
            max_tokens=llm_config.get('max_tokens'),
            temperature=llm_config.get('temperature'),
            client_type=llm_config.get('client_type'),
            base_url=llm_config.get('base_url')
        )
        logger.debug(f"Call to LLM API completed")

        # Save summary as markdown
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"# Summary: {file_name}\n\n")
            f.write(summary)

        logger.info(f"Summary saved: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Error in summarize_transcript: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    try:
        config = get_config()
        transcript_path = input("Enter the path to the transcript file: ")
        summarize_transcript(transcript_path, config)
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")