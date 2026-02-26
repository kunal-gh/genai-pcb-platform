"""
Unit tests for training configuration management.

Tests cover:
- YAML configuration loading
- Environment variable overrides
- Hyperparameter validation
- Default configuration usage
- Error handling
"""

import os
import pytest
import tempfile
from pathlib import Path
import yaml

from src.training.training_config import (
    TrainingConfig,
    load_training_config,
    VALIDATION_RANGES
)


class TestTrainingConfigDefaults:
    """Test default configuration values."""
    
    def test_default_config_creation(self):
        """Test that default config can be created without arguments."""
        config = TrainingConfig()
        
        assert config.batch_size == 32
        assert config.learning_rate == 1e-3
        assert config.hidden_dim == 256
        assert config.num_layers == 4
        assert config.num_epochs == 100
    
    def test_default_config_validation(self):
        """Test that default config passes validation."""
        # Should not raise any exceptions
        config = TrainingConfig()
        config._validate_hyperparameters()


class TestYAMLLoading:
    """Test YAML configuration file loading."""
    
    def test_load_from_yaml(self):
        """Test loading configuration from YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_content = {
                'batch_size': 64,
                'learning_rate': 0.0005,
                'hidden_dim': 512,
                'num_epochs': 50,
            }
            yaml.dump(yaml_content, f)
            temp_path = f.name
        
        try:
            config = TrainingConfig.from_yaml(temp_path)
            
            assert config.batch_size == 64
            assert config.learning_rate == 0.0005
            assert config.hidden_dim == 512
            assert config.num_epochs == 50
            
            # Check that unspecified values use defaults
            assert config.num_layers == 4
            assert config.dropout == 0.1
        finally:
            os.unlink(temp_path)
    
    def test_load_from_nonexistent_yaml(self):
        """Test that loading from nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            TrainingConfig.from_yaml("nonexistent_config.yaml")
    
    def test_load_from_empty_yaml(self):
        """Test loading from empty YAML file uses defaults."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")  # Empty file
            temp_path = f.name
        
        try:
            config = TrainingConfig.from_yaml(temp_path)
            
            # Should use all defaults
            assert config.batch_size == 32
            assert config.learning_rate == 1e-3
            assert config.hidden_dim == 256
        finally:
            os.unlink(temp_path)
    
    def test_save_to_yaml(self):
        """Test saving configuration to YAML file."""
        config = TrainingConfig(
            batch_size=128,
            learning_rate=0.002,
            hidden_dim=384
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "config.yaml"
            config.to_yaml(output_path)
            
            assert output_path.exists()
            
            # Load back and verify
            loaded_config = TrainingConfig.from_yaml(output_path)
            assert loaded_config.batch_size == 128
            assert loaded_config.learning_rate == 0.002
            assert loaded_config.hidden_dim == 384


class TestEnvironmentVariableOverrides:
    """Test environment variable overrides."""
    
    def test_env_override_int(self):
        """Test environment variable override for integer parameter."""
        os.environ['FALCON_BATCH_SIZE'] = '128'
        
        try:
            config = TrainingConfig.from_yaml_with_overrides()
            assert config.batch_size == 128
        finally:
            del os.environ['FALCON_BATCH_SIZE']
    
    def test_env_override_float(self):
        """Test environment variable override for float parameter."""
        os.environ['FALCON_LEARNING_RATE'] = '0.0005'
        
        try:
            config = TrainingConfig.from_yaml_with_overrides()
            assert config.learning_rate == 0.0005
        finally:
            del os.environ['FALCON_LEARNING_RATE']
    
    def test_env_override_bool(self):
        """Test environment variable override for boolean parameter."""
        os.environ['FALCON_USE_AUGMENTATION'] = 'false'
        
        try:
            config = TrainingConfig.from_yaml_with_overrides()
            assert config.use_augmentation is False
        finally:
            del os.environ['FALCON_USE_AUGMENTATION']
    
    def test_env_override_string(self):
        """Test environment variable override for string parameter."""
        os.environ['FALCON_DATASET_PATH'] = '/custom/path/to/data'
        
        try:
            config = TrainingConfig.from_yaml_with_overrides()
            assert config.dataset_path == '/custom/path/to/data'
        finally:
            del os.environ['FALCON_DATASET_PATH']
    
    def test_env_override_multiple(self):
        """Test multiple environment variable overrides."""
        os.environ['FALCON_BATCH_SIZE'] = '64'
        os.environ['FALCON_LEARNING_RATE'] = '0.002'
        os.environ['FALCON_HIDDEN_DIM'] = '512'
        
        try:
            config = TrainingConfig.from_yaml_with_overrides()
            assert config.batch_size == 64
            assert config.learning_rate == 0.002
            assert config.hidden_dim == 512
        finally:
            del os.environ['FALCON_BATCH_SIZE']
            del os.environ['FALCON_LEARNING_RATE']
            del os.environ['FALCON_HIDDEN_DIM']
    
    def test_env_override_with_yaml(self):
        """Test that environment variables override YAML values."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_content = {
                'batch_size': 32,
                'learning_rate': 0.001,
            }
            yaml.dump(yaml_content, f)
            temp_path = f.name
        
        os.environ['FALCON_BATCH_SIZE'] = '128'
        
        try:
            config = TrainingConfig.from_yaml_with_overrides(temp_path)
            
            # Environment variable should override YAML
            assert config.batch_size == 128
            # YAML value should be used for non-overridden params
            assert config.learning_rate == 0.001
        finally:
            del os.environ['FALCON_BATCH_SIZE']
            os.unlink(temp_path)
    
    def test_env_override_invalid_value(self):
        """Test that invalid environment variable values are ignored."""
        os.environ['FALCON_BATCH_SIZE'] = 'not_a_number'
        
        try:
            config = TrainingConfig.from_yaml_with_overrides()
            # Should use default value since conversion failed
            assert config.batch_size == 32
        finally:
            del os.environ['FALCON_BATCH_SIZE']


class TestHyperparameterValidation:
    """Test hyperparameter validation."""
    
    def test_valid_hyperparameters(self):
        """Test that valid hyperparameters pass validation."""
        config = TrainingConfig(
            batch_size=64,
            learning_rate=0.001,
            hidden_dim=256,
            num_layers=4,
            dropout=0.1
        )
        # Should not raise any exceptions
        config._validate_hyperparameters()
    
    def test_batch_size_too_small(self):
        """Test that batch_size below minimum raises ValueError."""
        with pytest.raises(ValueError, match="batch_size.*outside acceptable range"):
            TrainingConfig(batch_size=0)
    
    def test_batch_size_too_large(self):
        """Test that batch_size above maximum raises ValueError."""
        with pytest.raises(ValueError, match="batch_size.*outside acceptable range"):
            TrainingConfig(batch_size=300)
    
    def test_learning_rate_too_small(self):
        """Test that learning_rate below minimum raises ValueError."""
        with pytest.raises(ValueError, match="learning_rate.*outside acceptable range"):
            TrainingConfig(learning_rate=1e-7)
    
    def test_learning_rate_too_large(self):
        """Test that learning_rate above maximum raises ValueError."""
        with pytest.raises(ValueError, match="learning_rate.*outside acceptable range"):
            TrainingConfig(learning_rate=1.0)
    
    def test_dropout_too_small(self):
        """Test that dropout below minimum raises ValueError."""
        with pytest.raises(ValueError, match="dropout.*outside acceptable range"):
            TrainingConfig(dropout=-0.1)
    
    def test_dropout_too_large(self):
        """Test that dropout above maximum raises ValueError."""
        with pytest.raises(ValueError, match="dropout.*outside acceptable range"):
            TrainingConfig(dropout=1.0)
    
    def test_hidden_dim_too_small(self):
        """Test that hidden_dim below minimum raises ValueError."""
        with pytest.raises(ValueError, match="hidden_dim.*outside acceptable range"):
            TrainingConfig(hidden_dim=16)
    
    def test_num_layers_too_small(self):
        """Test that num_layers below minimum raises ValueError."""
        with pytest.raises(ValueError, match="num_layers.*outside acceptable range"):
            TrainingConfig(num_layers=0)
    
    def test_validation_error_message_format(self):
        """Test that validation error messages include parameter name and range."""
        try:
            TrainingConfig(batch_size=500)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            error_msg = str(e)
            assert "batch_size" in error_msg
            assert "500" in error_msg
            assert "[1, 256]" in error_msg


class TestConfigurationLoading:
    """Test the main load_training_config function."""
    
    def test_load_with_defaults(self):
        """Test loading configuration with defaults."""
        config = load_training_config()
        
        assert config.batch_size == 32
        assert config.learning_rate == 1e-3
        assert config.hidden_dim == 256
    
    def test_load_from_yaml_file(self):
        """Test loading configuration from YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_content = {
                'batch_size': 64,
                'learning_rate': 0.0005,
            }
            yaml.dump(yaml_content, f)
            temp_path = f.name
        
        try:
            config = load_training_config(temp_path)
            assert config.batch_size == 64
            assert config.learning_rate == 0.0005
        finally:
            os.unlink(temp_path)
    
    def test_load_with_missing_file_uses_defaults(self):
        """Test that missing config file falls back to defaults."""
        config = load_training_config("nonexistent_file.yaml")
        
        # Should use defaults without raising error
        assert config.batch_size == 32
        assert config.learning_rate == 1e-3
    
    def test_load_with_env_overrides(self):
        """Test loading with environment variable overrides."""
        os.environ['FALCON_BATCH_SIZE'] = '128'
        os.environ['FALCON_LEARNING_RATE'] = '0.002'
        
        try:
            config = load_training_config()
            assert config.batch_size == 128
            assert config.learning_rate == 0.002
        finally:
            del os.environ['FALCON_BATCH_SIZE']
            del os.environ['FALCON_LEARNING_RATE']


class TestConfigurationDictConversion:
    """Test configuration to dictionary conversion."""
    
    def test_to_dict(self):
        """Test converting configuration to dictionary."""
        config = TrainingConfig(
            batch_size=64,
            learning_rate=0.001,
            hidden_dim=512
        )
        
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['batch_size'] == 64
        assert config_dict['learning_rate'] == 0.001
        assert config_dict['hidden_dim'] == 512
        assert 'num_layers' in config_dict
        assert 'dropout' in config_dict


class TestConfigurationLogging:
    """Test configuration logging functionality."""
    
    def test_log_hyperparameters(self, caplog):
        """Test that log_hyperparameters logs all parameters."""
        config = TrainingConfig()
        
        with caplog.at_level('INFO'):
            config.log_hyperparameters()
        
        log_text = caplog.text
        
        # Check that key parameters are logged
        assert 'batch_size' in log_text
        assert 'learning_rate' in log_text
        assert 'hidden_dim' in log_text
        assert 'num_layers' in log_text
        assert 'Active Training Configuration' in log_text


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
