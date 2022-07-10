from pathlib import Path

import pytest
from deepdiff import DeepDiff
from pydantic import ValidationError

from classification_model.config.core import (
    create_and_validate_config,
    fetch_config_from_yaml,
)

TEST_CONFIG_TEXT = """
package_name: classification_model
data_url: https://www.openml.org/data/get_csv/16826755/phpMYEkMl
target: survived
pipeline_name: classification_model
pipeline_save_file: classification_model_output_v
drop_features:
 - name
 - ticket
 - boat
 - body
 - home_dest
numerical_vars:
 - age
 - fare
title_feature: title
extract_title: name
transform_vars:
 - cabin
categorical_vars:
 - sex
 - cabin
 - embarked
 - title
variables_to_rename:
 home.dest: home_dest
test_size: 0.2
random_state: 42
rare_label_tol: 0.05
alpha: 0.005
"""

INVALID_TEST_CONFIG_TEXT = """
package_name: classification_model
data_url: https://www.openml.org/data/get_csv/16826755/phpMYEkMl
target: survived
pipeline_name: classification_model
pipeline_save_file: classification_model_output_v
drop_features:
 - name
 - ticket
 - boat
 - body
 - home_dest
numerical_vars:
 - age
 - fare
title_feature: title
extract_title: name
transform_vars:
 - cabin
categorical_vars:
 - sex
 - cabin
 - embarked
variables_to_rename:
 home.dest: home_dest
test_size: 0.2
random_state: 42
rare_label_tol: 0.05
alpha: 0.005
"""


def test_fetch_config_structure(tmpdir):
    # Given
    # We make use of the pytest built-n tmpdir fixture
    configs_dir = Path(tmpdir)
    config_1 = configs_dir / "sample_config.yml"
    config_1.write_text(TEST_CONFIG_TEXT)
    parsed_config = fetch_config_from_yaml(cfg_path=config_1)

    # When
    config = create_and_validate_config(parsed_config=parsed_config)

    # Then
    assert config.model_config
    assert config.app_config


def test_config_compare_valid_and_invalid_config(tmpdir):
    # Given
    # We make use of the pytest built-in tmpdir fixture
    configs_dir = Path(tmpdir)
    valid_config = configs_dir / "sample_vlaid_config.yml"
    invalid_config = configs_dir / "sample_invalid_config.yml"

    # difference between valid and invalid config is only
    # 'title' which was missing under 'categorical_vars' list
    valid_config.write_text(TEST_CONFIG_TEXT)
    parsed_valid_config = fetch_config_from_yaml(cfg_path=valid_config)
    invalid_config.write_text(INVALID_TEST_CONFIG_TEXT)
    parsed_invalid_config = fetch_config_from_yaml(cfg_path=invalid_config)

    # When
    isDifferent = DeepDiff(
        parsed_valid_config.data, parsed_invalid_config.data, ignore_order=True
    )

    # Then
    assert "title" in str(isDifferent)


def test_missing_config_field_raises_validation_error(tmpdir):
    # Given
    # We make use of the pytest built-in tmpdir fixture
    configs_dir = Path(tmpdir)
    config_1 = configs_dir / "sample_config.yml"
    TEST_CONFIG_TEXT = """
    package_name: classification_model
    data_url: https://www.openml.org/data/get_csv/16826755/phpMYEkMl
    """
    config_1.write_text(TEST_CONFIG_TEXT)
    parsed_config = fetch_config_from_yaml(cfg_path=config_1)
    # When
    with pytest.raises(ValidationError) as excinfo:
        create_and_validate_config(parsed_config=parsed_config)

    # Then
    assert "field required" in str(excinfo.value)
    assert "pipeline_name" in str(excinfo.value)
