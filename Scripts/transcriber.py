
import os
import logging
from .transcriber_utils import transcribe_audio

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def transcribe_audio_flow(audio_file_path, output_folder, config):
    """
    Orchestrates the transcription process using the selected engine.
    """
    try:
        logger.info(f"Starting transcription for: {audio_file_path}")
        transcript_path = transcribe_audio(audio_file_path, output_folder, config)
        if transcript_path:
            logger.info(f"Transcription successful: {transcript_path}")
            return transcript_path
        else:
            logger.error(f"Transcription failed for: {audio_file_path}")
            return None
    except Exception as e:
        logger.error(f"Error in transcribe_audio_flow: {str(e)}")
        raise