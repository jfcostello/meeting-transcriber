import os
import whisper
import torch
import numpy as np
from whisper.audio import SAMPLE_RATE, N_FRAMES, HOP_LENGTH, pad_or_trim

def transcribe_audio(audio_file_path, output_folder, config):
    file_name = os.path.splitext(os.path.basename(audio_file_path))[0]
    output_path = os.path.join(output_folder, f"{file_name}_transcript.md")

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    cuda_enabled = torch.backends.cuda.is_built()

    print(f"CUDA available: {cuda_available}")
    print(f"PyTorch built with CUDA: {cuda_enabled}")

    # Determine the device to use
    device = config['device']
    if device == "auto":
        device = "cuda" if cuda_available and cuda_enabled else "cpu"
    elif device == "cuda" and not (cuda_available and cuda_enabled):
        print("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"
    
    print(f"Using device: {device}")

    # Load Whisper model
    model = whisper.load_model(config['model']).to(device)

    # Determine batch size
    if config['batch_size'] == "auto":
        batch_size = 16 if device == "cuda" else 4
    else:
        batch_size = config['batch_size']

    # Determine FP16 usage
    use_fp16 = config['use_fp16']
    if use_fp16 == "auto":
        use_fp16 = device == "cuda"

    # Determine segment length
    if config['segment_length'] == "auto":
        segment_length = 30  # Default to 30 seconds
    else:
        segment_length = config['segment_length']

    # Set up options
    options = whisper.DecodingOptions(
        language=config['language'] if config['language'] != "auto" else None,
        fp16=use_fp16 and device == "cuda"
    )

    try:
        # Load audio
        audio = whisper.load_audio(audio_file_path)
        
        # Process audio in segments
        segment_length_samples = segment_length * SAMPLE_RATE
        segments = [audio[i:i+segment_length_samples] for i in range(0, len(audio), segment_length_samples)]

        full_transcript = []
        for i in range(0, len(segments), batch_size):
            print(f"Processing batch {i//batch_size + 1}/{(len(segments)-1)//batch_size + 1}")
            batch = segments[i:i+batch_size]
            mels = [log_mel_spectrogram(segment, device=device) for segment in batch]
            
            for j, mel in enumerate(mels):
                result = whisper.decode(model, mel, options)
                full_transcript.append(result.text)

        # Save transcript as markdown
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"# Transcript: {file_name}\n\n")
            f.write(" ".join(full_transcript))

        print(f"Transcript saved: {output_path}")
        return output_path
    except Exception as e:
        print(f"Error processing {file_name}: {str(e)}")
        return None

def log_mel_spectrogram(audio, device="cpu"):
    """
    Compute the log-Mel spectrogram of an audio signal.
    """
    audio = pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(device)
    return mel