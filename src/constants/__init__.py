from pathlib import Path

# Constants to define the paths of configuration files.
# These paths are centralized here for easy management and future modifications.

# Path to the main configuration file that contains project settings and other configurations.
CONFIG_FILE_PATH = Path("src/config/config.yaml")

# Path to the parameters file which may contain hyperparameters, model training settings, etc.
PARAMS_FILE_PATH = Path("src/params.yaml")

# Path to the schema file which may contain data schema definitions, validation rules, etc.
SCHEMA_FILE_PATH = Path("src/schema.yaml")