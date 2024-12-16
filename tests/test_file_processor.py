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
        # BDD:
        #   Scenario: Add timestamp to filename (enabled)
        #     Given a filename and a config with timestamping enabled
        #     When the add_timestamp_to_filename function is called
        #     Then the function should return the filename with a timestamp prefix
        # Pass Criteria:
        #   The function returns the filename with a timestamp prefix.
        config = self.test_config.copy()
        config['add_timestamp'] = True
        filename = "test_file.mp4"
        timestamped_filename = add_timestamp_to_filename(filename, config)
        self.assertTrue(timestamped_filename.startswith(datetime.now().strftime("%Y-%m-%d-%H-%M-")))
        self.assertTrue(timestamped_filename.endswith(filename))

    def test_add_timestamp_to_filename_disabled(self):
        # BDD:
        #   Scenario: Add timestamp to filename (disabled)
        #     Given a filename and a config with timestamping disabled
        #     When the add_timestamp_to_filename function is called
        #     Then the function should return the original filename
        # Pass Criteria:
        #   The function returns the original filename without a timestamp prefix.
        config = self.test_config.copy()
        filename = "test_file.mp4"
        timestamped_filename = add_timestamp_to_filename(filename, config)
        self.assertEqual(timestamped_filename, filename)

    def test_process_videos_successful(self):
        # BDD:
        #   Scenario: Successful processing of a video file
        #     Given a video file in the queue folder
        #     When the process_videos function is called
        #     Then the audio should be extracted
        #     And the video file should be moved to the output folder
        #     And the audio file should be transcribed
        #     And the transcript should be summarized
        #     And the summary should be saved in the output folder
        # Pass Criteria:
        #   The video file, audio file, transcript, and summary are created in the output folder.
        config = self.test_config.copy()
        process_videos(self.test_queue_folder, config)
        self.assertTrue(os.path.exists(os.path.join(self.test_output_folder, "test_video.mp4")))
        self.assertTrue(os.path.exists(os.path.join(self.test_output_folder, "test_video.wav")))
        self.assertTrue(os.path.exists(os.path.join(self.test_output_folder, "test_video_transcript.md")))
        self.assertTrue(os.path.exists(os.path.join(self.test_output_folder, "test_video_summary.md")))

    def test_process_videos_error(self):
        # BDD:
        #   Scenario: Error during video processing
        #     Given a video file in the queue folder
        #     When the process_videos function is called and an error occurs during audio extraction
        #     Then the video file should be moved to the output folder
        # Pass Criteria:
        #   The video file is moved to the output folder even if an error occurs during audio extraction.
        config = self.test_config.copy()
        os.remove(self.test_video_file)
        with open(self.test_video_file, "w") as f:
            f.write("This is a dummy video file.")
        process_videos(self.test_queue_folder, config)
        self.assertTrue(os.path.exists(os.path.join(self.test_output_folder, "test_video.mp4")))

    def test_process_audio_files_successful(self):
        # BDD:
        #   Scenario: Successful processing of an audio file
        #     Given an audio file in the queue folder
        #     When the process_audio_files function is called
        #     Then the audio file should be transcribed
        #     And the transcript should be summarized
        #     And the audio file should be moved to the output folder
        #     And the summary should be saved in the output folder
        # Pass Criteria:
        #   The audio file, transcript, and summary are created in the output folder.
        config = self.test_config.copy()
        os.makedirs(os.path.join(self.test_queue_folder, "test_audio_dir"), exist_ok=True)
        shutil.move(self.test_audio_file, os.path.join(self.test_queue_folder, "test_audio_dir", "test_audio.mp3"))
        shutil.move(self.test_summary_rules_file, os.path.join(self.test_queue_folder, "test_audio_dir", "summary-rules.txt"))
        process_audio_files(self.test_queue_folder, config)
        self.assertTrue(os.path.exists(os.path.join(self.test_output_folder, "test_audio_dir", "test_audio.mp3")))
        self.assertTrue(os.path.exists(os.path.join(self.test_output_folder, "test_audio_dir", "test_audio_transcript.md")))
        self.assertTrue(os.path.exists(os.path.join(self.test_output_folder, "test_audio_dir", "test_audio_summary.md")))

    def test_process_audio_files_no_summary_rules(self):
        # BDD:
        #   Scenario: Processing an audio file without summary rules
        #     Given an audio file in the queue folder without a summary-rules.txt file
        #     When the process_audio_files function is called
        #     Then a warning message should be logged
        #     And the audio file should be transcribed
        #     And the transcript should be summarized using default settings
        #     And the audio file should be moved to the output folder
        #     And the summary should be saved in the output folder
        # Pass Criteria:
        #   The audio file, transcript, and summary are created in the output folder, and a warning message is logged.
        config = self.test_config.copy()
        os.makedirs(os.path.join(self.test_queue_folder, "test_audio_dir"), exist_ok=True)
        shutil.move(self.test_audio_file, os.path.join(self.test_queue_folder, "test_audio_dir", "test_audio.mp3"))
        process_audio_files(self.test_queue_folder, config)
        self.assertTrue(os.path.exists(os.path.join(self.test_output_folder, "test_audio_dir", "test_audio.mp3")))
        self.assertTrue(os.path.exists(os.path.join(self.test_output_folder, "test_audio_dir", "test_audio_transcript.md")))
        self.assertTrue(os.path.exists(os.path.join(self.test_output_folder, "test_audio_dir", "test_audio_summary.md")))

    def test_process_transcripts_successful(self):
        # BDD:
        #   Scenario: Successful processing of a transcript file
        #     Given a transcript file in the queue folder
        #     When the process_transcripts function is called
        #     Then the transcript should be summarized
        #     And the transcript file should be moved to the output folder
        #     And the summary should be saved in the output folder
        # Pass Criteria:
        #   The transcript file and summary are created in the output folder.
        config = self.test_config.copy()
        os.makedirs(os.path.join(self.test_queue_folder, "test_transcript_dir"), exist_ok=True)
        shutil.move(self.test_transcript_file, os.path.join(self.test_queue_folder, "test_transcript_dir", "test_transcript_transcript.md"))
        shutil.move(self.test_summary_rules_file, os.path.join(self.test_queue_folder, "test_transcript_dir", "summary-rules.txt"))
        process_transcripts(self.test_queue_folder, config)
        self.assertTrue(os.path.exists(os.path.join(self.test_output_folder, "test_transcript_dir", "test_transcript_transcript.md")))
        self.assertTrue(os.path.exists(os.path.join(self.test_output_folder, "test_transcript_dir", "test_transcript_summary.md")))

    def test_process_transcripts_no_summary_rules(self):
        # BDD:
        #   Scenario: Processing a transcript file without summary rules
        #     Given a transcript file in the queue folder without a summary-rules.txt file
        #     When the process_transcripts function is called
        #     Then a warning message should be logged
        #     And the transcript should be summarized using default settings
        #     And the transcript file should be moved to the output folder
        #     And the summary should be saved in the output folder
        # Pass Criteria:
        #   The transcript file and summary are created in the output folder, and a warning message is logged.
        config = self.test_config.copy()
        os.makedirs(os.path.join(self.test_queue_folder, "test_transcript_dir"), exist_ok=True)
        shutil.move(self.test_transcript_file, os.path.join(self.test_queue_folder, "test_transcript_dir", "test_transcript_transcript.md"))
        process_transcripts(self.test_queue_folder, config)
        self.assertTrue(os.path.exists(os.path.join(self.test_output_folder, "test_transcript_dir", "test_transcript_transcript.md")))
        self.assertTrue(os.path.exists(os.path.join(self.test_output_folder, "test_transcript_dir", "test_transcript_summary.md")))

if __name__ == '__main__':
    unittest.main()
