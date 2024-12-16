import unittest
from Scripts.audio_extractor import extract_audio
import os
import subprocess

class TestAudioExtractor(unittest.TestCase):
    def setUp(self):
        # Create a dummy video file for testing
        self.test_video_file = "test_video.mp4"
        # Create a valid dummy video file
        subprocess.run([
            "ffmpeg",
            "-f", "lavfi",
            "-i", "testsrc=duration=1:size=640x480:rate=30",
            "-f", "lavfi",
            "-i", "sine=frequency=440:duration=1",
            "-vcodec", "libx264",
            "-acodec", "aac",
            "-pix_fmt", "yuv420p",
            self.test_video_file
        ], check=True, capture_output=True, text=True)
        self.output_folder = "test_output"
        os.makedirs(self.output_folder, exist_ok=True)
        self.output_path = os.path.join(self.output_folder, "test_video.wav")

    def tearDown(self):
        # Clean up the dummy files and folders
        os.remove(self.test_video_file)
        if os.path.exists(self.output_path):
            os.remove(self.output_path)
        os.rmdir(self.output_folder)

    def test_extract_audio_successful(self):
        # BDD:
        #   Scenario: Successful audio extraction
        #     Given a valid video file
        #     When the extract_audio function is called
        #     Then the function should return the path to the extracted audio file
        #     And the audio file should exist
        # Pass Criteria:
        #   The function returns a valid path to an audio file, and the audio file exists at that path.
        audio_path = extract_audio(self.test_video_file, self.output_folder)
        self.assertTrue(os.path.exists(audio_path))

    def test_extract_audio_error(self):
        # BDD:
        #   Scenario: Error during audio extraction
        #     Given an invalid video file
        #     When the extract_audio function is called
        #     Then the function should raise an exception
        # Pass Criteria:
        #   The function raises an exception when called with an invalid video file.
        with self.assertRaises(Exception):
            extract_audio("non_existent_file.mp4", self.output_folder)

if __name__ == '__main__':
    unittest.main()
