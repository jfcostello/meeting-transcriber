import os
import logging
from whisper.audio import SAMPLE_RATE, pad_or_trim
import whisper
import torch
from faster_whisper import WhisperModel
import whisperx

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
    logger.info(f"Whisper model dimensions: {model.dims}")

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

            # Log the language
            language = config.get('language', "auto")
            logger.info(f"Transcribe language: {language}")

            # Transcribe the segment
            result = model.transcribe(segment, language=language)

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
        
        # New configuration option for VAD
        trim_silence = config.get('trim_silence', False)
        # Convert string "true" to boolean True, everything else to False
        vad_filter = str(trim_silence).lower() == "true"

        model = WhisperModel(model_size, device=device, compute_type=compute_type)

        # Use the vad_filter parameter in the transcribe method
        segments, info = model.transcribe(audio_file_path, beam_size=beam_size, vad_filter=vad_filter)

        transcript_text = ""
        for segment in segments:
            logger.info(f"Segment {segment.id}: {segment.text}")
            transcript_text += segment.text + " "

        # Save transcript as markdown
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(transcript_text)

        logger.info(f"Transcript saved: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error processing {file_name} with Faster Whisper: {str(e)}")
        return None

def transcribe_with_whisperx(audio_file_path, output_folder, config):
    """
    Transcribe audio using WhisperX for word-level timestamps and optional speaker diarization.
    """
    file_name = os.path.splitext(os.path.basename(audio_file_path))[0]
    output_path = os.path.join(output_folder, f"{file_name}_transcript.md")

    try:
        model_name = config.get("model", "large-v2")
        device = config.get("device", "auto")
        compute_type = config.get("compute_type", "float16")
        language = config.get("language", "en")
        batch_size = config.get("batch_size", 16)
        
        cuda_available = torch.cuda.is_available()
        cuda_enabled = torch.backends.cudnn.enabled and torch.backends.cuda.is_built()
        if device == "auto":
            device = "cuda" if cuda_available and cuda_enabled else "cpu"
        elif device == "cuda" and not (cuda_available and cuda_enabled):
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            device = "cpu"

        diarize = str(config.get("diarize", "false")).lower() == "true"
        hf_token = config.get("hf_token", None)
        min_speakers = config.get("min_speakers", None)
        max_speakers = config.get("max_speakers", None)
        return_char_alignments = str(config.get("return_char_alignments", "false")).lower() == "true"
        # Removed 'vad_filter' usage here, since it's not supported in the Python function:
        # vad_filter = str(config.get("vad_filter", "true")).lower() == "true"
        highlight_words = str(config.get("highlight_words", "false")).lower() == "true"

        logger.info(f"Using WhisperX model: {model_name} on device: {device}")
        
        # 1) Transcribe (faster-whisper backend) in WhisperX
        model = whisperx.load_model(model_name, device, compute_type=compute_type)
        audio = whisperx.load_audio(audio_file_path)
        logger.info(f"WhisperX transcribing with batch_size={batch_size}")

        # Remove 'vad_filter=vad_filter' from here
        result = model.transcribe(
            audio, 
            batch_size=batch_size,
            language=language,
        )
        
        # 2) Align with wav2vec2
        logger.info(f"Loading alignment model for language '{result['language']}'")
        align_model, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(
            result["segments"], 
            align_model, 
            metadata, 
            audio, 
            device, 
            return_char_alignments=return_char_alignments
        )

        # 3) Optional speaker diarization
        if diarize:
            if not hf_token:
                logger.warning("Diarization requested but no hf_token supplied. Diarization will fail.")
            else:
                logger.info(f"Running speaker diarization with token: {hf_token}")
                diarize_model = whisperx.DiarizationPipeline(
                    use_auth_token=hf_token, 
                    device=device
                )
                diarize_segments = diarize_model(
                    audio,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers
                )
                result = whisperx.assign_word_speakers(diarize_segments, result)

        # Combine text for writing to a single markdown file
        transcript_text = ""
        for seg in result["segments"]:
            speaker_label = ""
            if diarize and "speaker" in seg:
                speaker_label = f"Speaker {seg['speaker']}: "
            transcript_text += f"{speaker_label}{seg['text']} "

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(transcript_text.strip())

        logger.info(f"WhisperX transcript saved: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error processing {file_name} with WhisperX: {str(e)}")
        return None


def transcribe_audio(audio_file_path, output_folder, config):
    """
    Select and execute the appropriate transcription engine based on configuration.
    """
    engine = config.get('transcription_engine', 'whisper')
    output_folder = os.path.dirname(audio_file_path)

    if engine == 'whisper':
        return transcribe_with_whisper(audio_file_path, output_folder, config.get('whisper', {}))
    elif engine == 'faster_whisper':
        return transcribe_with_faster_whisper(audio_file_path, output_folder, config.get('faster_whisper', {}))
    elif engine == 'whisperx':
        return transcribe_with_whisperx(audio_file_path, output_folder, config.get('whisperx', {}))
    else:
        raise ValueError(f"Unsupported transcription engine: {engine}")

