import unittest
import os
import shutil
from datetime import datetime
from Scripts.file_processor import process_videos, process_audio_files, process_transcripts, add_timestamp_to_filename
from Scripts.config_handler import get_config

class TestFileProcessor(unittest.TestCase):
    def setUp(self):
        # Create dummy directories and files for testing
        self.test_queue_folder = "test_queue"
        os.makedirs(self.test_queue_folder, exist_ok=True)
        self.test_output_folder = "test_output"
        os.makedirs(self.test_output_folder, exist_ok=True)
        self.test_video_file = os.path.join(self.test_queue_folder, "test_video.mp4")
        self.test_audio_file = os.path.join(self.test_queue_folder, "test_audio.mp3")
        self.test_transcript_file = os.path.join(self.test_queue_folder, "test_transcript_transcript.md")
        self.test_summary_rules_file = os.path.join(self.test_queue_folder, "summary-rules.txt")
        
        with open(self.test_video_file, "w") as f:
            f.write("This is a dummy video file.")
        with open(self.test_audio_file, "w") as f:
            f.write("This is a dummy audio file.")
        with open(self.test_transcript_file, "w") as f:
            f.write("This is a dummy transcript file.")
        with open(self.test_summary_rules_file, "w") as f:
            f.write("This is a dummy summary rules file.")
        
        # Create a dummy config
        self.test_config = {
            'meeting_recordings_folder': self.test_queue_folder,
            'output_structure': {
                'base_folder': self.test_output_folder,
                'structure': []
            },
            'add_timestamp': False,
            'logging': {'enabled': False}
        }

    def tearDown(self):
        # Clean up the dummy files and folders
        shutil.rmtree(self.test_queue_folder)
        shutil.rmtree(self.test_output_folder)

    def test_add_timestamp_to_filename_enabled(self):
        # BDD: Scenario: Processing a video file with timestamping enabled - Then the video file is renamed with a timestamp
        # Test: Checks if the filename is correctly timestamped when timestamping is enabled
        config = self.test_config.copy()
        config['add_timestamp'] = True
        filename = "test_file.mp4"
        timestamped_filename = add_timestamp_to_filename(filename, config)
        self.assertTrue(timestamped_filename.startswith(datetime.now().strftime("%Y-%m-%d-%H-%M-")))
        self.assertTrue(timestamped_filename.endswith(filename))

    def test_add_timestamp_to_filename_disabled(self):
        # BDD: Scenario: Processing a video file with timestamping disabled - Then the video file is not renamed with a timestamp
        # Test: Checks if the filename is not timestamped when timestamping is disabled
        config = self.test_config.copy()
        filename = "test_file.mp4"
        timestamped_filename = add_timestamp_to_filename(filename, config)
        self.assertEqual(timestamped_filename, filename)

    def test_process_videos_successful(self):
        # BDD: Scenario: Successful processing of a video file - Then the audio is extracted, the video file is moved to the output folder, the audio file is transcribed, the transcript is summarized, and the summary is saved in the output folder
        # Test: Checks if a video file is processed successfully
        config = self.test_config.copy()
        process_videos(self.test_queue_folder, config)
        self.assertTrue(os.path.exists(os.path.join(self.test_output_folder, "test_video.mp4")))
        self.assertTrue(os.path.exists(os.path.join(self.test_output_folder, "test_video.wav")))
        self.assertTrue(os.path.exists(os.path.join(self.test_output_folder, "test_video_transcript.md")))
        self.assertTrue(os.path.exists(os.path.join(self.test_output_folder, "test_video_summary.md")))

    def test_process_videos_error(self):
        # BDD: Scenario: Processing a video file with an error during audio extraction - Then an error message is logged and the video file is moved to the output folder
        # Test: Checks if an error is handled correctly during video processing
        config = self.test_config.copy()
        os.remove(self.test_video_file)
        with open(self.test_video_file, "w") as f:
            f.write("This is a dummy video file.")
        process_videos(self.test_queue_folder, config)
        self.assertTrue(os.path.exists(os.path.join(self.test_output_folder, "test_video.mp4")))

    def test_process_audio_files_successful(self):
        # BDD: Scenario: Successful processing of an audio file - Then the audio file is transcribed, the transcript is summarized, the audio file is moved to the output folder, and the summary is saved in the output folder
        # Test: Checks if an audio file is processed successfully
        config = self.test_config.copy()
        os.makedirs(os.path.join(self.test_queue_folder, "test_audio_dir"), exist_ok=True)
        shutil.move(self.test_audio_file, os.path.join(self.test_queue_folder, "test_audio_dir", "test_audio.mp3"))
        shutil.move(self.test_summary_rules_file, os.path.join(self.test_queue_folder, "test_audio_dir", "summary-rules.txt"))
        process_audio_files(self.test_queue_folder, config)
        self.assertTrue(os.path.exists(os.path.join(self.test_output_folder, "test_audio_dir", "test_audio.mp3")))
        self.assertTrue(os.path.exists(os.path.join(self.test_output_folder, "test_audio_dir", "test_audio_transcript.md")))
        self.assertTrue(os.path.exists(os.path.join(self.test_output_folder, "test_audio_dir", "test_audio_summary.md")))

    def test_process_audio_files_no_summary_rules(self):
        # BDD: Scenario: Processing an audio file without a summary-rules.txt file - Then a warning message is logged, the audio file is transcribed, the transcript is summarized using default settings, the audio file is moved to the output folder, and the summary is saved in the output folder
        # Test: Checks if an audio file is processed correctly when no summary rules are provided
        config = self.test_config.copy()
        os.makedirs(os.path.join(self.test_queue_folder, "test_audio_dir"), exist_ok=True)
        shutil.move(self.test_audio_file, os.path.join(self.test_queue_folder, "test_audio_dir", "test_audio.mp3"))
        process_audio_files(self.test_queue_folder, config)
        self.assertTrue(os.path.exists(os.path.join(self.test_output_folder, "test_audio_dir", "test_audio.mp3")))
        self.assertTrue(os.path.exists(os.path.join(self.test_output_folder, "test_audio_dir", "test_audio_transcript.md")))
        self.assertTrue(os.path.exists(os.path.join(self.test_output_folder, "test_audio_dir", "test_audio_summary.md")))

    def test_process_transcripts_successful(self):
        # BDD: Scenario: Successful processing of a transcript file - Then the transcript is summarized, the transcript file is moved to the output folder, and the summary is saved in the output folder
        # Test: Checks if a transcript file is processed successfully
        config = self.test_config.copy()
        os.makedirs(os.path.join(self.test_queue_folder, "test_transcript_dir"), exist_ok=True)
        shutil.move(self.test_transcript_file, os.path.join(self.test_queue_folder, "test_transcript_dir", "test_transcript_transcript.md"))
        shutil.move(self.test_summary_rules_file, os.path.join(self.test_queue_folder, "test_transcript_dir", "summary-rules.txt"))
        process_transcripts(self.test_queue_folder, config)
        self.assertTrue(os.path.exists(os.path.join(self.test_output_folder, "test_transcript_dir", "test_transcript_transcript.md")))
        self.assertTrue(os.path.exists(os.path.join(self.test_output_folder, "test_transcript_dir", "test_transcript_summary.md")))

    def test_process_transcripts_no_summary_rules(self):
        # BDD: Scenario: Processing a transcript file without a summary-rules.txt file - Then a warning message is logged, the transcript is summarized using default settings, the transcript file is moved to the output folder, and the summary is saved in the output folder
        # Test: Checks if a transcript file is processed correctly when no summary rules are provided
        config = self.test_config.copy()
        os.makedirs(os.path.join(self.test_queue_folder, "test_transcript_dir"), exist_ok=True)
        shutil.move(self.test_transcript_file, os.path.join(self.test_queue_folder, "test_transcript_dir", "test_transcript_transcript.md"))
        process_transcripts(self.test_queue_folder, config)
        self.assertTrue(os.path.exists(os.path.join(self.test_output_folder, "test_transcript_dir", "test_transcript_transcript.md")))
        self.assertTrue(os.path.exists(os.path.join(self.test_output_folder, "test_transcript_dir", "test_transcript_summary.md")))

if __name__ == '__main__':
    unittest.main()
