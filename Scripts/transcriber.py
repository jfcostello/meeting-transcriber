import os
import whisper

def transcribe_audio(audio_file_path, output_folder, model_name, language=None):
    file_name = os.path.splitext(os.path.basename(audio_file_path))[0]
    output_path = os.path.join(output_folder, f"{file_name}_transcript.md")

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Load Whisper model
    model = whisper.load_model(model_name)

    # Transcribe audio
    result = model.transcribe(audio_file_path, language=language)

    # Save transcript as markdown
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# Transcript: {file_name}\n\n")
        f.write(result["text"])

    print(f"Transcript saved: {output_path}")
    return output_path