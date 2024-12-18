import os
import logging
from .transcriber_utils import transcribe_audio
from .config_handler import get_config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def transcribe_audio_flow(audio_file_path, queue_folder, config):
    """
    Orchestrates the transcription process using the selected engine.
    """
    log_enabled = get_config().get('logging', {}).get('enabled', False)
    if log_enabled:
        logger.info(f"transcribe_audio_flow: Starting transcription for: {audio_file_path}")
        logger.info(f"transcribe_audio_flow: Loaded config: {config}")
        logger.info(f"transcribe_audio_flow: Using transcription engine: {config.get('transcription_engine')}")
        if config.get('transcription_engine') == 'whisper':
            logger.info(f"transcribe_audio_flow: Using whisper model: {config.get('whisper', {}).get('model')}")
        elif config.get('transcription_engine') == 'faster_whisper':
            logger.info(f"transcribe_audio_flow: Using faster_whisper model: {config.get('faster_whisper', {}).get('model')}")
    try:
        output_folder = os.path.dirname(audio_file_path)
        transcript_path = transcribe_audio(audio_file_path, output_folder, config)
        if transcript_path:
            if log_enabled:
                logger.info(f"transcribe_audio_flow: Transcription successful: {transcript_path}")
            return transcript_path
        else:
            logger.error(f"Transcription failed for: {audio_file_path}")
            return None
    except Exception as e:
        logger.error(f"Error in transcribe_audio_flow: {str(e)}")
        raise
