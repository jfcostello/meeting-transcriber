import unittest
import os
from unittest.mock import patch, MagicMock
from Scripts.transcriber_utils import transcribe_with_whisper, transcribe_with_faster_whisper, transcribe_audio

class TestTranscriberUtils(unittest.TestCase):
    def setUp(self):
        # Create dummy files and folders for testing
        self.test_audio_folder = "test_audio"
        os.makedirs(self.test_audio_folder, exist_ok=True)
        self.test_audio_file = os.path.join(self.test_audio_folder, "test_audio.mp3")
        with open(self.test_audio_file, "w") as f:
            f.write("This is a test audio file.")
        self.test_config = {
            'whisper': {'model': 'test_whisper_model', 'language': 'en'},
            'faster_whisper': {'model': 'test_faster_whisper_model', 'device': 'cpu', 'compute_type': 'int8'},
            'transcription_engine': 'whisper'
        }

    def tearDown(self):
        # Clean up the dummy files and folders
        os.remove(self.test_audio_file)
        os.rmdir(self.test_audio_folder)

    @patch('Scripts.transcriber_utils.whisper.load_model')
    @patch('Scripts.transcriber_utils.whisper.load_audio')
    def test_transcribe_with_whisper_successful(self, mock_load_audio, mock_load_model):
        # BDD:
        #   Scenario: Successful transcription with Whisper
        #     Given an audio file and whisper config
        #     When the transcribe_with_whisper function is called
        #     Then the audio file should be transcribed using whisper
        #     And the function should return the path to the transcript file
        # Pass Criteria:
        #   The audio file is transcribed using whisper, and the function returns the path to the transcript file.
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        mock_result = {"text": "This is a test whisper transcript."}
        mock_model.transcribe.return_value = mock_result
        mock_load_audio.return_value = [1, 2, 3]
        
        config = self.test_config.copy()
        transcript_path = transcribe_with_whisper(self.test_audio_file, self.test_audio_folder, config['whisper'])
        self.assertTrue(os.path.exists(transcript_path))
        with open(transcript_path, "r") as f:
            transcript = f.read()
        self.assertEqual(transcript, "This is a test whisper transcript.")
        mock_model.transcribe.assert_called_once()

    @patch('Scripts.transcriber_utils.WhisperModel')
    def test_transcribe_with_faster_whisper_successful(self, mock_whisper_model):
        # BDD:
        #   Scenario: Successful transcription with Faster Whisper
        #     Given an audio file and faster whisper config
        #     When the transcribe_with_faster_whisper function is called
        #     Then the audio file should be transcribed using faster whisper
        #     And the function should return the path to the transcript file
        # Pass Criteria:
        #   The audio file is transcribed using faster whisper, and the function returns the path to the transcript file.
        mock_model = MagicMock()
        mock_whisper_model.return_value = mock_model
        mock_segments = [MagicMock(text="This is a test faster whisper transcript.", id=1)]
        mock_model.transcribe.return_value = (mock_segments, {})
        
        config = self.test_config.copy()
        transcript_path = transcribe_with_faster_whisper(self.test_audio_file, self.test_audio_folder, config['faster_whisper'])
        self.assertTrue(os.path.exists(transcript_path))
        with open(transcript_path, "r") as f:
            transcript = f.read()
        self.assertEqual(transcript, "This is a test faster whisper transcript. ")
        mock_model.transcribe.assert_called_once()

    def test_transcribe_with_whisper_error(self):
        # BDD:
        #   Scenario: Error during Whisper transcription
        #     Given an audio file and whisper config
        #     When the transcribe_with_whisper function is called and an error occurs
        #     Then the function should raise an exception
        # Pass Criteria:
        #   The function raises an exception when an error occurs during whisper transcription.
        config = self.test_config.copy()
        with self.assertRaises(Exception):
            transcribe_with_whisper("non_existent_file.mp3", self.test_audio_folder, config['whisper'])

    def test_transcribe_with_faster_whisper_error(self):
        # BDD:
        #   Scenario: Error during Faster Whisper transcription
        #     Given an audio file and faster whisper config
        #     When the transcribe_with_faster_whisper function is called and an error occurs
        #     Then the function should raise an exception
        # Pass Criteria:
        #   The function raises an exception when an error occurs during faster whisper transcription.
        config = self.test_config.copy()
        with self.assertRaises(Exception):
            transcribe_with_faster_whisper("non_existent_file.mp3", self.test_audio_folder, config['faster_whisper'])

    @patch('Scripts.transcriber_utils.transcribe_with_whisper')
    @patch('Scripts.transcriber_utils.transcribe_with_faster_whisper')
    def test_transcribe_audio_selects_whisper(self, mock_faster_whisper, mock_whisper):
        # BDD:
        #   Scenario: Select Whisper transcription engine
        #     Given an audio file and a config with 'whisper' as the transcription engine
        #     When the transcribe_audio function is called
        #     Then the whisper transcription engine should be selected
        # Pass Criteria:
        #   The whisper transcription engine is selected.
        config = self.test_config.copy()
        config['transcription_engine'] = 'whisper'
        transcribe_audio(self.test_audio_file, self.test_audio_folder, config)
        mock_whisper.assert_called_once()
        mock_faster_whisper.assert_not_called()

    @patch('Scripts.transcriber_utils.transcribe_with_whisper')
    @patch('Scripts.transcriber_utils.transcribe_with_faster_whisper')
    def test_transcribe_audio_selects_faster_whisper(self, mock_faster_whisper, mock_whisper):
        # BDD:
        #   Scenario: Select Faster Whisper transcription engine
        #     Given an audio file and a config with 'faster_whisper' as the transcription engine
        #     When the transcribe_audio function is called
        #     Then the faster whisper transcription engine should be selected
        # Pass Criteria:
        #   The faster whisper transcription engine is selected.
        config = self.test_config.copy()
        config['transcription_engine'] = 'faster_whisper'
        transcribe_audio(self.test_audio_file, self.test_audio_folder, config)
        mock_faster_whisper.assert_called_once()
        mock_whisper.assert_not_called()

    def test_transcribe_audio_unsupported_engine(self):
        # BDD:
        #   Scenario: Unsupported transcription engine
        #     Given an audio file and a config with an unsupported transcription engine
        #     When the transcribe_audio function is called
        #     Then the function should raise a ValueError
        # Pass Criteria:
        #   The function raises a ValueError when an unsupported transcription engine is specified.
        config = self.test_config.copy()
        config['transcription_engine'] = 'unsupported'
        with self.assertRaises(ValueError):
            transcribe_audio(self.test_audio_file, self.test_audio_folder, config)

if __name__ == '__main__':
    unittest.main()
