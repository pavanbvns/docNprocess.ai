# config_loader.py
import yaml
import logging
import os
import pyfiglet
from rich.console import Console


def load_config(config_file_path="config.yml"):
    """Loads configuration settings from the specified YAML file."""
    try:
        with open(config_file_path, "r") as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        logging.error(f"Config file not found at {config_file_path}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing config file: {e}")
        raise


def setup_logging(log_file_path="logs/app.log"):
    """Sets up logging with a specified log file path."""

    if not os.path.exists(os.path.dirname(log_file_path)):
        os.makedirs(os.path.dirname(log_file_path))
        # Create logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler(),  # Adds logging to the console
        ],
    )

    return logging.getLogger(__name__)


def display_banner():
    console = Console()
    # Use pyfiglet to create ASCII art
    banner = pyfiglet.figlet_format("ot-synapse.ai", font="standard")
    # Print banner with color using rich
    console.print(banner, style="bold blue")
