import typing as t
from pathlib import Path

from pydantic import BaseModel
from strictyaml import YAML, load

import classification_model

# Project Directories
PACKAGE_ROOT = Path(classification_model.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"


class AppConfig(BaseModel):
    """
    Application-level config.
    """

    package_name: str
    pipeline_name: str
    data_url: str
    pipeline_save_file: str


class ModelConfig(BaseModel):
    """
    All configuration relevant to model training and feature engineering.
    """

    target: str
    drop_features: t.List[str]
    numerical_vars: t.List[str]
    categorical_vars: t.List[str]
    test_size: float
    random_state: int
    extract_title: str
    transform_vars: t.List[str]
    rare_label_tol: float
    alpha: float
    title_feature: str
    variables_to_rename: t.Dict


class Config(BaseModel):
    """
    Master config object.
    """

    app_config: AppConfig
    model_config: ModelConfig


def find_config_file() -> Path:
    """
    Locate the configuration file.
    Returns:
        Path: file config path
    """
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH}")


def fetch_config_from_yaml(cfg_path: Path = None) -> YAML:
    """Parse YAML containing the package confiuration."""

    if cfg_path is None:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # specify the data attribute from the strictyaml YAML type.
    _config = Config(
        app_config=AppConfig(**parsed_config.data),
        model_config=ModelConfig(**parsed_config.data),
    )

    return _config


config = create_and_validate_config()
