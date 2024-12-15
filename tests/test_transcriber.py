import unittest
import os
from unittest.mock import patch
from Scripts.transcriber import transcribe_audio_flow
from Scripts.config_handler import get_config

class TestTranscriber(unittest.TestCase):
    def setUp(self):
        # Create dummy files and folders for testing
        self.test_audio_folder = "test_audio"
        os.makedirs(self.test_audio_folder, exist_ok=True)
        self.test_audio_file = os.path.join(self.test_audio_folder, "test_audio.mp3")
        with open(self.test_audio_file, "w") as f:
            f.write("This is a test audio file.")
        self.test_config = {
            'transcription_engine': 'test_engine',
            'whisper': {'model': 'test_whisper_model'},
            'faster_whisper': {'model': 'test_faster_whisper_model'},
            'logging': {'enabled': False}
        }

    def tearDown(self):
        # Clean up the dummy files and folders
        os.remove(self.test_audio_file)
        os.rmdir(self.test_audio_folder)

    @patch('Scripts.transcriber.transcribe_audio')
    def test_transcribe_audio_flow_successful(self, mock_transcribe_audio):
        # BDD: Scenario: Successful processing of an audio file - Then the audio file is transcribed
        # Test: Checks if the transcription flow is executed successfully
        mock_transcribe_audio.return_value = os.path.join(self.test_audio_folder, "test_audio_transcript.md")
        config = self.test_config.copy()
        transcript_path = transcribe_audio_flow(self.test_audio_file, self.test_audio_folder, config)
        self.assertEqual(transcript_path, os.path.join(self.test_audio_folder, "test_audio_transcript.md"))
        mock_transcribe_audio.assert_called_once()

    @patch('Scripts.transcriber.transcribe_audio')
    def test_transcribe_audio_flow_error(self, mock_transcribe_audio):
        # BDD: Scenario: Processing an audio file with an error during transcription - Then an error message is logged
        # Test: Checks if an error is handled correctly during transcription
        mock_transcribe_audio.side_effect = Exception("Transcription error")
        config = self.test_config.copy()
        with self.assertRaises(Exception):
            transcribe_audio_flow(self.test_audio_file, self.test_audio_folder, config)
        mock_transcribe_audio.assert_called_once()

if __name__ == '__main__':
    unittest.main()
