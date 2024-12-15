import unittest
import os
import shutil
from datetime import datetime
from Scripts.utils import move_file
from Scripts.config_handler import get_config

class TestUtils(unittest.TestCase):
    def setUp(self):
        # Create dummy files and folders for testing
        self.test_source_folder = "test_source"
        os.makedirs(self.test_source_folder, exist_ok=True)
        self.test_file = os.path.join(self.test_source_folder, "test_file.txt")
        with open(self.test_file, "w") as f:
            f.write("This is a test file.")
        self.test_config = {
            'output_structure': {
                'base_folder': 'test_output',
                'structure': []
            },
            'logging': {'enabled': False}
        }

    def tearDown(self):
        # Clean up the dummy files and folders
        shutil.rmtree(self.test_source_folder)
        shutil.rmtree("test_output")

    def test_move_file_no_structure(self):
        # BDD: Scenario: Using different output structures - Then the output file is saved in the specified output structure
        # Test: Checks if the file is moved correctly when no output structure is specified
        config = self.test_config.copy()
        move_file(self.test_file, config)
        self.assertTrue(os.path.exists(os.path.join("test_output", "test_file.txt")))

    def test_move_file_with_date_structure(self):
        # BDD: Scenario: Using different output structures - Then the output file is saved in the specified output structure
        # Test: Checks if the file is moved correctly when the output structure includes date
        config = self.test_config.copy()
        config['output_structure']['structure'] = ['DATE']
        move_file(self.test_file, config)
        today_date = datetime.now().strftime("%Y-%m-%d")
        self.assertTrue(os.path.exists(os.path.join("test_output", today_date, "test_file.txt")))

    def test_move_file_with_filename_structure(self):
        # BDD: Scenario: Using different output structures - Then the output file is saved in the specified output structure
        # Test: Checks if the file is moved correctly when the output structure includes filename
        config = self.test_config.copy()
        config['output_structure']['structure'] = ['FILE-NAME']
        move_file(self.test_file, config)
        self.assertTrue(os.path.exists(os.path.join("test_output", "test_file", "test_file.txt")))

    def test_move_file_with_summary_type_structure(self):
        # BDD: Scenario: Using different output structures - Then the output file is saved in the specified output structure
        # Test: Checks if the file is moved correctly when the output structure includes summary type
        config = self.test_config.copy()
        config['output_structure']['structure'] = ['SUMMARY-TYPE']
        os.makedirs(os.path.join(self.test_source_folder, "test_summary_type"), exist_ok=True)
        shutil.move(self.test_file, os.path.join(self.test_source_folder, "test_summary_type", "test_file.txt"))
        test_file = os.path.join(self.test_source_folder, "test_summary_type", "test_file.txt")
        move_file(test_file, config)
        self.assertTrue(os.path.exists(os.path.join("test_output", "test_summary_type", "test_file.txt")))

    def test_move_file_with_all_structures(self):
        # BDD: Scenario: Using different output structures - Then the output file is saved in the specified output structure
        # Test: Checks if the file is moved correctly when all output structures are included
        config = self.test_config.copy()
        config['output_structure']['structure'] = ['DATE', 'SUMMARY-TYPE', 'FILE-NAME']
        os.makedirs(os.path.join(self.test_source_folder, "test_summary_type"), exist_ok=True)
        shutil.move(self.test_file, os.path.join(self.test_source_folder, "test_summary_type", "test_file.txt"))
        test_file = os.path.join(self.test_source_folder, "test_summary_type", "test_file.txt")
        move_file(test_file, config)
        today_date = datetime.now().strftime("%Y-%m-%d")
        self.assertTrue(os.path.exists(os.path.join("test_output", today_date, "test_summary_type", "test_file", "test_file.txt")))

if __name__ == '__main__':
    unittest.main()
