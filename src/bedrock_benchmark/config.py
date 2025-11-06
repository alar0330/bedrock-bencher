"""
Configuration management for the Bedrock Benchmark Toolkit.

Provides centralized configuration handling with support for:
- Configuration files (YAML/JSON)
- Environment variables
- Command-line overrides
- Validation and defaults
"""

import os
import json
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Any, Union
import logging

from pydantic import BaseModel, Field, validator


logger = logging.getLogger(__name__)


class BenchmarkConfig(BaseModel):
    """
    Main configuration class for the Bedrock Benchmark Toolkit.
    
    Supports loading from files, environment variables, and validation
    of all configuration parameters.
    """
    
    # AWS Configuration
    aws_region: str = Field(default="us-east-1", description="AWS region for Bedrock service")
    aws_profile: Optional[str] = Field(default=None, description="AWS profile name")
    
    # Concurrency Configuration
    max_concurrent: int = Field(default=10, ge=1, le=100, description="Maximum concurrent requests")
    
    # Retry Configuration
    max_retries: int = Field(default=5, ge=0, le=20, description="Maximum retry attempts")
    base_delay: float = Field(default=1.0, ge=0.1, le=10.0, description="Base delay for exponential backoff (seconds)")
    max_delay: float = Field(default=60.0, ge=1.0, le=300.0, description="Maximum delay for exponential backoff (seconds)")
    
    # Storage Configuration
    storage_path: str = Field(default="./experiments", description="Path to store experiment data")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="structured", description="Log format: 'structured' or 'simple'")
    log_file: Optional[str] = Field(default=None, description="Optional log file path")
    
    # Model Configuration Defaults
    default_model_params: Dict[str, Any] = Field(default_factory=dict, description="Default model parameters")
    
    @validator('aws_region')
    def validate_aws_region(cls, v):
        """Validate AWS region format."""
        if not v or len(v) < 3:
            raise ValueError("AWS region must be a valid region identifier")
        return v
    
    @validator('log_level')
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
    
    @validator('log_format')
    def validate_log_format(cls, v):
        """Validate log format."""
        valid_formats = ['structured', 'simple']
        if v not in valid_formats:
            raise ValueError(f"Log format must be one of: {valid_formats}")
        return v
    
    @validator('storage_path')
    def validate_storage_path(cls, v):
        """Validate and normalize storage path."""
        path = Path(v).expanduser().resolve()
        return str(path)
    
    @validator('max_delay')
    def validate_max_delay_greater_than_base(cls, v, values):
        """Ensure max_delay is greater than base_delay."""
        if 'base_delay' in values and v <= values['base_delay']:
            raise ValueError("max_delay must be greater than base_delay")
        return v
    
    class Config:
        """Pydantic configuration."""
        extra = "forbid"  # Don't allow extra fields
        validate_assignment = True  # Validate on assignment


class ConfigManager:
    """
    Manages configuration loading from multiple sources with precedence:
    1. Command-line arguments (highest priority)
    2. Environment variables
    3. Configuration file
    4. Defaults (lowest priority)
    """
    
    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Optional path to configuration file
        """
        self.config_file = Path(config_file) if config_file else None
        self._config: Optional[BenchmarkConfig] = None
    
    def load_config(
        self,
        config_overrides: Optional[Dict[str, Any]] = None,
        validate: bool = True
    ) -> BenchmarkConfig:
        """
        Load configuration from all sources with proper precedence.
        
        Args:
            config_overrides: Dictionary of configuration overrides (highest priority)
            validate: Whether to validate the configuration
        
        Returns:
            BenchmarkConfig instance
        
        Raises:
            ValueError: If configuration is invalid
            FileNotFoundError: If specified config file doesn't exist
        """
        # Start with defaults
        config_data = {}
        
        # Load from configuration file if specified
        if self.config_file:
            file_config = self._load_config_file(self.config_file)
            config_data.update(file_config)
        
        # Load from environment variables
        env_config = self._load_from_environment()
        config_data.update(env_config)
        
        # Apply overrides (highest priority)
        if config_overrides:
            config_data.update(config_overrides)
        
        # Create and validate configuration
        try:
            self._config = BenchmarkConfig(**config_data)
            
            if validate:
                self._validate_config()
            
            logger.info(f"Configuration loaded successfully")
            logger.debug(f"Config: {self._config.dict()}")
            
            return self._config
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise ValueError(f"Invalid configuration: {e}")
    
    def _load_config_file(self, config_file: Path) -> Dict[str, Any]:
        """
        Load configuration from a file (JSON or YAML).
        
        Args:
            config_file: Path to configuration file
        
        Returns:
            Dictionary of configuration values
        
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is unsupported or invalid
        """
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        try:
            with open(config_file, 'r') as f:
                if config_file.suffix.lower() in ['.yaml', '.yml']:
                    return yaml.safe_load(f) or {}
                elif config_file.suffix.lower() == '.json':
                    return json.load(f) or {}
                else:
                    raise ValueError(f"Unsupported config file format: {config_file.suffix}")
        
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise ValueError(f"Invalid configuration file format: {e}")
        except Exception as e:
            raise ValueError(f"Error reading configuration file: {e}")
    
    def _load_from_environment(self) -> Dict[str, Any]:
        """
        Load configuration from environment variables.
        
        Environment variables should be prefixed with 'BEDROCK_BENCHMARK_'
        and use uppercase with underscores.
        
        Returns:
            Dictionary of configuration values from environment
        """
        env_config = {}
        prefix = "BEDROCK_BENCHMARK_"
        
        # Mapping of environment variable names to config field names
        env_mappings = {
            f"{prefix}AWS_REGION": "aws_region",
            f"{prefix}AWS_PROFILE": "aws_profile",
            f"{prefix}MAX_CONCURRENT": "max_concurrent",
            f"{prefix}MAX_RETRIES": "max_retries",
            f"{prefix}BASE_DELAY": "base_delay",
            f"{prefix}MAX_DELAY": "max_delay",
            f"{prefix}STORAGE_PATH": "storage_path",
            f"{prefix}LOG_LEVEL": "log_level",
            f"{prefix}LOG_FORMAT": "log_format",
            f"{prefix}LOG_FILE": "log_file",
        }
        
        for env_var, config_key in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert string values to appropriate types
                if config_key in ["max_concurrent", "max_retries"]:
                    try:
                        env_config[config_key] = int(value)
                    except ValueError:
                        logger.warning(f"Invalid integer value for {env_var}: {value}")
                elif config_key in ["base_delay", "max_delay"]:
                    try:
                        env_config[config_key] = float(value)
                    except ValueError:
                        logger.warning(f"Invalid float value for {env_var}: {value}")
                else:
                    env_config[config_key] = value
        
        # Handle AWS region from standard AWS environment variable
        if "aws_region" not in env_config:
            aws_region = os.getenv("AWS_DEFAULT_REGION") or os.getenv("AWS_REGION")
            if aws_region:
                env_config["aws_region"] = aws_region
        
        return env_config
    
    def _validate_config(self):
        """
        Perform additional validation beyond Pydantic validation.
        
        Raises:
            ValueError: If configuration is invalid
        """
        if not self._config:
            raise ValueError("No configuration loaded")
        
        # Validate storage path is writable
        storage_path = Path(self._config.storage_path)
        try:
            storage_path.mkdir(parents=True, exist_ok=True)
            # Test write access
            test_file = storage_path / ".write_test"
            test_file.touch()
            test_file.unlink()
        except Exception as e:
            raise ValueError(f"Storage path is not writable: {storage_path} - {e}")
        
        # Validate log file path if specified
        if self._config.log_file:
            log_file = Path(self._config.log_file)
            try:
                log_file.parent.mkdir(parents=True, exist_ok=True)
                # Test write access
                log_file.touch()
            except Exception as e:
                raise ValueError(f"Log file path is not writable: {log_file} - {e}")
    
    def get_config(self) -> Optional[BenchmarkConfig]:
        """
        Get the current configuration.
        
        Returns:
            BenchmarkConfig instance or None if not loaded
        """
        return self._config
    
    def save_config(self, output_path: Union[str, Path], format: str = "yaml"):
        """
        Save current configuration to a file.
        
        Args:
            output_path: Path to save configuration file
            format: Output format ('yaml' or 'json')
        
        Raises:
            ValueError: If no configuration is loaded or invalid format
        """
        if not self._config:
            raise ValueError("No configuration loaded to save")
        
        output_path = Path(output_path)
        config_dict = self._config.dict()
        
        try:
            with open(output_path, 'w') as f:
                if format.lower() == 'yaml':
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                elif format.lower() == 'json':
                    json.dump(config_dict, f, indent=2)
                else:
                    raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Configuration saved to {output_path}")
            
        except Exception as e:
            raise ValueError(f"Error saving configuration: {e}")


def load_default_config() -> BenchmarkConfig:
    """
    Load configuration with default settings and environment variables.
    
    Returns:
        BenchmarkConfig instance with defaults and environment overrides
    """
    config_manager = ConfigManager()
    return config_manager.load_config()


def find_config_file() -> Optional[Path]:
    """
    Find configuration file in standard locations.
    
    Searches for configuration files in the following order:
    1. ./bedrock-bencher.yaml
    2. ./bedrock-bencher.yml
    3. ./bedrock-bencher.json
    4. ~/.bedrock-bencher.yaml
    5. ~/.bedrock-bencher.yml
    6. ~/.bedrock-bencher.json
    
    Returns:
        Path to configuration file or None if not found
    """
    search_paths = [
        Path("./bedrock-bencher.yaml"),
        Path("./bedrock-bencher.yml"),
        Path("./bedrock-bencher.json"),
        Path("~/.bedrock-bencher.yaml").expanduser(),
        Path("~/.bedrock-bencher.yml").expanduser(),
        Path("~/.bedrock-bencher.json").expanduser(),
    ]
    
    for path in search_paths:
        if path.exists():
            logger.info(f"Found configuration file: {path}")
            return path
    
    return None


def load_config_with_auto_discovery(
    config_file: Optional[Union[str, Path]] = None,
    config_overrides: Optional[Dict[str, Any]] = None
) -> BenchmarkConfig:
    """
    Load configuration with automatic file discovery.
    
    Args:
        config_file: Explicit config file path (overrides auto-discovery)
        config_overrides: Configuration overrides
    
    Returns:
        BenchmarkConfig instance
    """
    # Use explicit config file or try to find one
    if config_file:
        config_path = Path(config_file)
    else:
        config_path = find_config_file()
    
    config_manager = ConfigManager(config_path)
    return config_manager.load_config(config_overrides)