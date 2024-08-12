import yaml
import os

config = None

def load_config():
    global config
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_config():
    global config
    if config is None:
        config = load_config()
    return config

def update_config(key, value):
    global config
    if config is None:
        config = load_config()
    config[key] = value
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
    with open(config_path, 'w') as file:
        yaml.dump(config, file)

def get_prompt_template(template_name):
    global config
    if config is None:
        config = load_config()
    return config['prompts'].get(template_name, "")