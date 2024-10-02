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

    try:
        # Load audio
        audio = whisper.load_audio(audio_file_path)
        
        # Define segment length (30 seconds)
        segment_length = 30 * whisper.audio.SAMPLE_RATE

        # Process audio in segments
        segments = [audio[i:i+segment_length] for i in range(0, len(audio), segment_length)]

        full_transcript = []
        for i, segment in enumerate(segments):
            print(f"\nProcessing segment {i+1}/{len(segments)}")
            
            # Pad or trim the segment
            segment = whisper.pad_or_trim(segment)
            
            # Transcribe the segment
            result = model.transcribe(segment, language=config['language'] if config['language'] != "auto" else None)
            
            full_transcript.append(result["text"])
            print(result["text"], flush=True)  # Print each segment as it's transcribed

        # Save transcript as markdown
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"# Transcript: {file_name}\n\n")
            f.write(" ".join(full_transcript))

        print(f"\nTranscript saved: {output_path}")
        return output_path
    except Exception as e:
        print(f"Error processing {file_name}: {str(e)}")
        return None