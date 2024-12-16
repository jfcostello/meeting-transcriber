import unittest
import os
import yaml
from Scripts.config_handler import load_config, get_summary_prompt, update_config, get_add_timestamp_config

class TestConfigHandler(unittest.TestCase):
    def setUp(self):
        os.environ['TEST_CONFIG'] = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test_config.yaml')

    def tearDown(self):
        if 'TEST_CONFIG' in os.environ:
            del os.environ['TEST_CONFIG']

    def test_load_config(self):
        # BDD:
        #   Scenario: Load configuration file
        #     Given the config.yaml file exists
        #     When the load_config function is called
        #     Then the function should return a dictionary
        #     And the dictionary should contain the 'llm' and 'transcription_engine' keys
        # Pass Criteria:
        #   The function returns a dictionary, and the dictionary contains the 'llm' and 'transcription_engine' keys.
        config = load_config()
        self.assertIsInstance(config, dict)
        self.assertIn('llm', config)
        self.assertIn('transcription_engine', config)

    def test_get_summary_prompt(self):
        # BDD:
        #   Scenario: Get summary prompt
        #     Given a config dictionary
        #     When the get_summary_prompt function is called
        #     Then if 'summary_type' or 'summary_type_presets_folder' are missing, a ValueError is raised
        #     And if the prompt file does not exist, a FileNotFoundError is raised
        #     And if the prompt file exists, the function should return the prompt content
        # Pass Criteria:
        #   The function raises ValueError if 'summary_type' or 'summary_type_presets_folder' are missing.
        #   The function raises FileNotFoundError if the prompt file does not exist.
        #   The function returns the prompt content if the prompt file exists.
        config = load_config()
        with self.assertRaises(ValueError):
            get_summary_prompt({})
        
        config['summary_type'] = 'test'
        config['summary_type_presets_folder'] = 'test_presets'
        
        with self.assertRaises(FileNotFoundError):
            get_summary_prompt(config)
        
        # Create a dummy prompt file
        os.makedirs(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test_presets'), exist_ok=True)
        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test_presets', 'test.txt'), 'w') as f:
            f.write('This is a test prompt.')
        
        prompt = get_summary_prompt(config)
        self.assertEqual(prompt, 'This is a test prompt.')
        
        # Clean up dummy prompt file
        os.remove(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test_presets', 'test.txt'))
        os.rmdir(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test_presets'))

    def test_update_config(self):
        # BDD:
        #   Scenario: Update configuration
        #     Given a config dictionary
        #     When the update_config function is called with a key and value
        #     Then the config file should be updated with the new value
        #     And if the key does not exist, a KeyError is raised
        # Pass Criteria:
        #   The config file is updated with the new value.
        #   The function raises KeyError if the key does not exist.
        config = load_config()
        original_llm_model = config['llm']['model']
        
        update_config('llm', {'model': 'new_model'})
        
        updated_config = load_config()
        self.assertEqual(updated_config['llm']['model'], 'new_model')
        
        update_config('llm', {'model': original_llm_model}) # Reset the config
        
        updated_config = load_config()
        self.assertEqual(updated_config['llm']['model'], original_llm_model)
        
        with self.assertRaises(KeyError):
            update_config('non_existent_key', 'some_value')

    def test_get_add_timestamp_config(self):
        # BDD:
        #   Scenario: Get add timestamp config
        #     Given a config dictionary
        #     When the get_add_timestamp_config function is called
        #     Then the function should return the value of the 'add_timestamp' key
        # Pass Criteria:
        #   The function returns the value of the 'add_timestamp' key.
        
        config = load_config()
        add_timestamp = get_add_timestamp_config()
        self.assertEqual(add_timestamp, config.get('add_timestamp', False))

if __name__ == '__main__':
    unittest.main()