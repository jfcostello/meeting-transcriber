import os
import subprocess

def extract_audio(video_file_path, output_folder):
    file_name = os.path.splitext(os.path.basename(video_file_path))[0]
    output_path = os.path.join(output_folder, f"{file_name}.wav")

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Use ffmpeg to extract audio
    try:
        subprocess.run([
            "ffmpeg",
            "-i", video_file_path,
            "-acodec", "pcm_s16le",
            "-ac", "1",
            "-ar", "16000",
            output_path
        ], check=True, capture_output=True, text=True)
        print(f"Audio extracted: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio: {e}")
        print(f"ffmpeg stderr: {e.stderr}")
        raise