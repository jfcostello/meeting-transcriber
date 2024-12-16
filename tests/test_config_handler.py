import unittest
import os
import yaml
from unittest.mock import patch, mock_open
from Scripts.config_handler import load_config, get_config, get_summary_prompt, update_config, get_add_timestamp_config

class TestConfigHandler(unittest.TestCase):
    def setUp(self):
        # Create a dummy config file for testing
        self.test_config_path = "test_config.yaml"
        os.environ['TEST_CONFIG'] = self.test_config_path
        self.test_config_content = {
            'meeting_recordings_folder': 'test_recordings',
            'output_structure': {
                'base_folder': 'test_summaries',
                'structure': ['DATE', 'SUMMARY-TYPE', 'FILE-NAME']
            },
            'logging': {'enabled': True},
            'llm': {
                'model': 'test_model',
                'client_type': 'test_client',
                'max_tokens': 1000,
                'temperature': 0.5
            },
            'add_timestamp': True,
            'transcription_engine': 'test_engine',
            'summary_type': 'test_summary',
            'summary_type_presets_folder': 'test_presets'
        }
        with open(self.test_config_path, 'w') as f:
            yaml.dump(self.test_config_content, f)
        
        # Create a dummy summary preset file
        self.test_presets_folder = "test_presets"
        os.makedirs(self.test_presets_folder, exist_ok=True)
        self.test_summary_preset_path = os.path.join(self.test_presets_folder, "test_summary.txt")
        with open(self.test_summary_preset_path, 'w') as f:
            f.write("This is a test summary prompt.")

    def tearDown(self):
        # Clean up the dummy files
        os.remove(self.test_config_path)
        os.remove(self.test_summary_preset_path)
        os.rmdir(self.test_presets_folder)

    def test_load_config_successful(self):
        # BDD: N/A - Internal function
        # Test: Checks if the config file is loaded successfully
        with patch('Scripts.config_handler.open', mock_open(read_data=yaml.dump(self.test_config_content))) as mock_file:
            config = load_config()
            self.assertIsInstance(config, dict)
            self.assertEqual(config['meeting_recordings_folder'], 'test_recordings')
            self.assertEqual(config['output_structure']['base_folder'], 'test_summaries')
            self.assertEqual(config['logging']['enabled'], True)
            self.assertEqual(config['llm']['model'], 'test_model')
            self.assertEqual(config['add_timestamp'], True)
            self.assertEqual(config['transcription_engine'], 'test_engine')
            self.assertEqual(config['summary_type'], 'test_summary')

    def test_load_config_error(self):
        # BDD: N/A - Internal function
        # Test: Checks if an error is raised when the config file is not found
        with patch('Scripts.config_handler.os.path.exists', return_value=False):
            with self.assertRaises(FileNotFoundError):
                load_config()

    def test_get_config_successful(self):
        # BDD: N/A - Internal function
        # Test: Checks if the config is retrieved successfully
        with patch('Scripts.config_handler.load_config', return_value=self.test_config_content):
            config = get_config()
            self.assertIsInstance(config, dict)
            self.assertEqual(config['meeting_recordings_folder'], 'test_recordings')
            self.assertEqual(config['output_structure']['base_folder'], 'test_summaries')
            self.assertEqual(config['logging']['enabled'], True)
            self.assertEqual(config['llm']['model'], 'test_model')
            self.assertEqual(config['add_timestamp'], True)
            self.assertEqual(config['transcription_engine'], 'test_engine')
            self.assertEqual(config['summary_type'], 'test_summary')

    @patch('Scripts.config_handler.get_config')
    def test_get_summary_prompt_successful(self, mock_get_config):
        # BDD: Scenario: Using different summarization settings - Then the transcript is summarized using the specified summarization settings
        # Test: Checks if the summary prompt is loaded successfully
        mock_get_config.return_value = self.test_config_content
        config = get_config()
        prompt = get_summary_prompt(config)
        self.assertEqual(prompt, "This is a test summary prompt.")

    @patch('Scripts.config_handler.get_config')
    def test_get_summary_prompt_error(self, mock_get_config):
        # BDD: Scenario: Processing an audio file without a summary-rules.txt file - Then a warning message is logged
        # Test: Checks if an error is raised when the summary preset file is not found
        mock_get_config.return_value = self.test_config_content
        config = get_config()
        config['summary_type'] = 'non_existent_summary'
        with self.assertRaises(Exception):
            get_summary_prompt(config)

    def test_update_config_successful(self):
        # BDD: N/A - Internal function
        # Test: Checks if the config is updated successfully
        with patch('Scripts.config_handler.load_config', return_value=self.test_config_content):
            update_config('add_timestamp', False)
            config = get_config()
            self.assertFalse(config['add_timestamp'])

    def test_update_config_error(self):
        # BDD: N/A - Internal function
        # Test: Checks if an error is raised when the config file cannot be updated
        with patch('Scripts.config_handler.load_config', return_value=self.test_config_content):
            with self.assertRaises(KeyError):
                update_config('non_existent_key', 'test_value')

    def test_get_add_timestamp_config_successful(self):
        # BDD: Scenario: Processing a video file with timestamping enabled - Then the video file is renamed with a timestamp
        # Test: Checks if the add_timestamp config is retrieved successfully
        with patch('Scripts.config_handler.load_config', return_value=self.test_config_content):
            add_timestamp = get_add_timestamp_config()
            self.assertEqual(add_timestamp, self.test_config_content['add_timestamp'])

if __name__ == '__main__':
    unittest.main()
