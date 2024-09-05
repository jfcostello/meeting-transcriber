import os
import yaml
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info("Config loaded successfully")
        return config
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        raise

def get_config():
    return load_config()

def get_summary_prompt(config):
    try:
        summary_type = config.get('summary_type')
        presets_folder = config.get('summary_type_presets_folder')
        
        if not summary_type:
            raise ValueError("'summary_type' not found in config")
        if not presets_folder:
            raise ValueError("'summary_type_presets_folder' not found in config")
        
        prompt_file = f"{summary_type}.txt"
        prompt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), presets_folder, prompt_file)
        
        if os.path.exists(prompt_path):
            with open(prompt_path, 'r') as file:
                prompt = file.read().strip()
            # Clean up the prompt: remove extra whitespace and newlines
            prompt = ' '.join(prompt.split())
            logger.info(f"Summary prompt loaded and cleaned: {prompt[:100]}...")  # Log first 100 chars
            return prompt
        else:
            raise FileNotFoundError(f"Summary type preset file not found: {prompt_path}")
    except Exception as e:
        logger.error(f"Error in get_summary_prompt: {str(e)}")
        raise

def update_config(key, value):
    config = load_config()
    config[key] = value
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
    try:
        with open(config_path, 'w') as file:
            yaml.dump(config, file)
        logger.info(f"Config updated: {key} = {value}")
    except Exception as e:
        logger.error(f"Error updating config: {str(e)}")
        raise

def get_add_timestamp_config():
    config = load_config()
    return config.get('add_timestamp', False)