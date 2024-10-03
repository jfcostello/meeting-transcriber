import os
import logging
from whisper.audio import SAMPLE_RATE, pad_or_trim
import whisper
import torch
from faster_whisper import WhisperModel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def transcribe_with_whisper(audio_file_path, output_folder, config):
    """
    Transcribe audio using OpenAI's Whisper model.
    """
    file_name = os.path.splitext(os.path.basename(audio_file_path))[0]
    output_path = os.path.join(output_folder, f"{file_name}_transcript.md")

    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    cuda_enabled = torch.backends.cudnn.enabled and torch.backends.cuda.is_built()

    logger.info(f"CUDA available: {cuda_available}")
    logger.info(f"PyTorch built with CUDA: {cuda_enabled}")

    # Determine the device to use
    device = config.get('device', 'auto')
    if device == "auto":
        device = "cuda" if cuda_available and cuda_enabled else "cpu"
    elif device == "cuda" and not (cuda_available and cuda_enabled):
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    logger.info(f"Using device: {device}")

    # Load Whisper model
    model = whisper.load_model(config.get('model', 'base')).to(device)

    try:
        # Load audio
        audio = whisper.load_audio(audio_file_path)

        # Define segment length (30 seconds)
        segment_length = 30 * SAMPLE_RATE

        # Process audio in segments
        segments = [audio[i:i+segment_length] for i in range(0, len(audio), segment_length)]

        full_transcript = []
        for i, segment in enumerate(segments):
            logger.info(f"Processing segment {i+1}/{len(segments)}")

            # Pad or trim the segment
            segment = pad_or_trim(segment)

            # Transcribe the segment
            result = model.transcribe(segment, language=config.get('language', "auto"))

            full_transcript.append(result["text"])
            logger.info(f"Segment {i+1} transcription: {result['text']}")  # Changed to info

        # Save transcript as markdown
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(" ".join(full_transcript))

        logger.info(f"Transcript saved: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error processing {file_name} with Whisper: {str(e)}")
        return None

def transcribe_with_faster_whisper(audio_file_path, output_folder, config):
    """
    Transcribe audio using Faster Whisper model.
    """
    file_name = os.path.splitext(os.path.basename(audio_file_path))[0]
    output_path = os.path.join(output_folder, f"{file_name}_transcript.md")

    try:
        model_size = config.get('model', 'base-v3')
        device = config.get('device', 'cuda')
        compute_type = config.get('compute_type', 'float16')
        beam_size = config.get('beam_size', 5)

        model = WhisperModel(model_size, device=device, compute_type=compute_type)

        segments, info = model.transcribe(audio_file_path, beam_size=beam_size)

        transcript_text = ""
        for i, segment in enumerate(segments):
            logger.info(f"Segment {i+1} transcription: {segment.text}")
            transcript_text += segment.text + " "

        # Save transcript as markdown
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(transcript_text)

        logger.info(f"Transcript saved: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error processing {file_name} with Faster Whisper: {str(e)}")
        return None

def transcribe_audio(audio_file_path, output_folder, config):
    """
    Select and execute the appropriate transcription engine based on configuration.
    """
    engine = config.get('transcription_engine', 'whisper')
    if engine == 'whisper':
        return transcribe_with_whisper(audio_file_path, output_folder, config.get('whisper', {}))
    elif engine == 'faster_whisper':
        return transcribe_with_faster_whisper(audio_file_path, output_folder, config.get('faster_whisper', {}))
    else:
        raise ValueError(f"Unsupported transcription engine: {engine}")