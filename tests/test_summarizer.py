import unittest
import os
from unittest.mock import patch
from Scripts.summarizer import summarize_transcript, get_unique_filename
from Scripts.config_handler import get_config

class TestSummarizer(unittest.TestCase):
    def setUp(self):
        # Create dummy files and folders for testing
        self.test_transcript_folder = "test_transcripts"
        os.makedirs(self.test_transcript_folder, exist_ok=True)
        self.test_transcript_file = os.path.join(self.test_transcript_folder, "test_transcript.md")
        with open(self.test_transcript_file, "w") as f:
            f.write("This is a test transcript.")
        self.test_summary_rules_file = os.path.join(self.test_transcript_folder, "summary-rules.txt")
        with open(self.test_summary_rules_file, "w") as f:
            f.write("This is a test summary rules file.")
        self.test_config = {
            'llm': {
                'model': 'test_model',
                'client_type': 'test_client',
                'max_tokens': 1000,
                'temperature': 0.5
            },
            'logging': {'enabled': False}
        }

    def tearDown(self):
        # Clean up the dummy files and folders
        os.remove(self.test_transcript_file)
        os.remove(self.test_summary_rules_file)
        os.remove(os.path.join(self.test_transcript_folder, "test_transcript_summary.md"))
        os.rmdir(self.test_transcript_folder)

    @patch('Scripts.summarizer.call_llm_api')
    def test_summarize_transcript_successful(self, mock_call_llm_api):
        # BDD: Scenario: Successful processing of a transcript file - Then the transcript is summarized and the summary is saved in the output folder
        # Test: Checks if the transcript is summarized successfully
        mock_call_llm_api.return_value = "This is a test summary."
        config = self.test_config.copy()
        summary_path = summarize_transcript(self.test_transcript_file, config)
        self.assertTrue(os.path.exists(summary_path))
        with open(summary_path, "r") as f:
            summary = f.read()
        self.assertEqual(summary, "This is a test summary.")

    @patch('Scripts.summarizer.call_llm_api')
    def test_summarize_transcript_no_summary_rules(self, mock_call_llm_api):
        # BDD: Scenario: Processing a transcript file without a summary-rules.txt file - Then a warning message is logged and the transcript is summarized using default settings
        # Test: Checks if the transcript is summarized successfully when no summary rules are provided
        mock_call_llm_api.return_value = "This is a test summary."
        config = self.test_config.copy()
        os.remove(self.test_summary_rules_file)
        summary_path = summarize_transcript(self.test_transcript_file, config)
        self.assertTrue(os.path.exists(summary_path))
        with open(summary_path, "r") as f:
            summary = f.read()
        self.assertEqual(summary, "This is a test summary.")

    def test_summarize_transcript_error(self):
        # BDD: Scenario: Processing a transcript file with an error during summarization - Then an error message is logged
        # Test: Checks if an error is raised when the LLM API call fails
        config = self.test_config.copy()
        with self.assertRaises(Exception):
            summarize_transcript("non_existent_file.md", config)

    def test_get_unique_filename(self):
        # BDD: N/A - Internal function
        # Test: Checks if the unique filename is generated correctly
        base_path = os.path.join(self.test_transcript_folder, "test_transcript_summary.md")
        unique_path = get_unique_filename(base_path)
        self.assertEqual(unique_path, base_path)
        
        # Create a file with the same name
        with open(base_path, "w") as f:
            f.write("This is a test summary.")
        
        unique_path = get_unique_filename(base_path)
        self.assertEqual(unique_path, os.path.join(self.test_transcript_folder, "test_transcript_summary_1.md"))
        os.remove(base_path)

if __name__ == '__main__':
    unittest.main()
