"""Unit tests for configuration file support."""

import pytest
import tempfile
from pathlib import Path
import yaml

from sfunctor.utils.config import (
    load_config,
    validate_config,
    create_config_template,
    _deep_merge,
    get_default_config_path,
)
from sfunctor.utils.cli import RunConfig


class TestConfigLoading:
    """Test configuration file loading."""
    
    def test_load_valid_config(self):
        """Test loading a valid configuration file."""
        config_content = """
analysis:
  stride: 4
  n_disp_total: 5000
  N_random_subsamples: 2000
  n_ell_bins: 64
  stencil_width: 3
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            temp_path = Path(f.name)
        
        try:
            config = load_config(temp_path)
            assert config['analysis']['stride'] == 4
            assert config['analysis']['n_disp_total'] == 5000
            assert config['analysis']['stencil_width'] == 3
        finally:
            temp_path.unlink()
    
    def test_load_config_with_profile(self):
        """Test loading configuration with profile."""
        config_content = """
analysis:
  stride: 2
  n_disp_total: 1000

profiles:
  test:
    analysis:
      stride: 8
      n_disp_total: 100
  production:
    analysis:
      stride: 1
      n_disp_total: 10000
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            temp_path = Path(f.name)
        
        try:
            # Load base config
            config = load_config(temp_path)
            assert config['analysis']['stride'] == 2
            
            # Load with test profile
            config = load_config(temp_path, profile='test')
            assert config['analysis']['stride'] == 8
            assert config['analysis']['n_disp_total'] == 100
            
            # Load with production profile
            config = load_config(temp_path, profile='production')
            assert config['analysis']['stride'] == 1
            assert config['analysis']['n_disp_total'] == 10000
        finally:
            temp_path.unlink()
    
    def test_load_nonexistent_file(self):
        """Test loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_file.yaml")
    
    def test_load_invalid_yaml(self):
        """Test loading invalid YAML raises error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_path = Path(f.name)
        
        try:
            with pytest.raises(yaml.YAMLError):
                load_config(temp_path)
        finally:
            temp_path.unlink()
    
    def test_nonexistent_profile(self):
        """Test loading nonexistent profile raises error."""
        config_content = """
analysis:
  stride: 2

profiles:
  test:
    analysis:
      stride: 8
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            temp_path = Path(f.name)
        
        try:
            with pytest.raises(KeyError) as exc_info:
                load_config(temp_path, profile='nonexistent')
            assert 'nonexistent' in str(exc_info.value)
        finally:
            temp_path.unlink()


class TestConfigValidation:
    """Test configuration validation."""
    
    def test_valid_config(self):
        """Test validation of valid configuration."""
        config = RunConfig(
            stride=2,
            file_name=Path("test.npz"),
            slice_list=None,
            n_disp_total=1000,
            N_random_subsamples=1000,
            n_ell_bins=128,
            n_processes=4,
            stencil_width=3
        )
        # Should not raise
        validate_config(config)
    
    def test_invalid_stride(self):
        """Test validation catches invalid stride."""
        config = RunConfig(
            stride=0,  # Invalid
            file_name=Path("test.npz"),
            slice_list=None,
            n_disp_total=1000,
            N_random_subsamples=1000,
            n_ell_bins=128,
            n_processes=4,
            stencil_width=2
        )
        with pytest.raises(ValueError) as exc_info:
            validate_config(config)
        assert "stride" in str(exc_info.value)
    
    def test_invalid_stencil_width(self):
        """Test validation catches invalid stencil width."""
        config = RunConfig(
            stride=2,
            file_name=Path("test.npz"),
            slice_list=None,
            n_disp_total=1000,
            N_random_subsamples=1000,
            n_ell_bins=128,
            n_processes=4,
            stencil_width=4  # Invalid - must be 2, 3, or 5
        )
        with pytest.raises(ValueError) as exc_info:
            validate_config(config)
        assert "stencil_width" in str(exc_info.value)
    
    def test_no_input_files(self):
        """Test validation catches missing input files."""
        config = RunConfig(
            stride=2,
            file_name=None,  # No file
            slice_list=None,  # No list
            n_disp_total=1000,
            N_random_subsamples=1000,
            n_ell_bins=128,
            n_processes=4,
            stencil_width=2
        )
        with pytest.raises(ValueError) as exc_info:
            validate_config(config)
        assert "file_name or slice_list" in str(exc_info.value)
    
    def test_both_input_files(self):
        """Test validation catches mutually exclusive inputs."""
        config = RunConfig(
            stride=2,
            file_name=Path("test.npz"),  # File specified
            slice_list=Path("list.txt"),  # List also specified
            n_disp_total=1000,
            N_random_subsamples=1000,
            n_ell_bins=128,
            n_processes=4,
            stencil_width=2
        )
        with pytest.raises(ValueError) as exc_info:
            validate_config(config)
        assert "mutually exclusive" in str(exc_info.value)


class TestConfigUtilities:
    """Test configuration utility functions."""
    
    def test_deep_merge(self):
        """Test deep dictionary merging."""
        base = {
            'a': 1,
            'b': {'x': 2, 'y': 3},
            'c': [1, 2, 3]
        }
        overlay = {
            'a': 10,
            'b': {'y': 30, 'z': 40},
            'd': 50
        }
        
        result = _deep_merge(base, overlay)
        
        assert result['a'] == 10  # Overridden
        assert result['b']['x'] == 2  # Preserved
        assert result['b']['y'] == 30  # Overridden
        assert result['b']['z'] == 40  # Added
        assert result['c'] == [1, 2, 3]  # Preserved
        assert result['d'] == 50  # Added
    
    def test_create_config_template(self):
        """Test config template generation."""
        template = create_config_template()
        
        # Should be valid YAML
        config = yaml.safe_load(template)
        
        # Check expected sections exist
        assert 'analysis' in config
        assert 'binning' in config
        assert 'output' in config
        assert 'physics' in config
        assert 'profiles' in config
        
        # Check some expected values
        assert config['analysis']['stride'] == 1
        assert config['profiles']['quick_test']['analysis']['n_disp_total'] == 100
    
    def test_get_default_config_path(self):
        """Test default config path detection."""
        # Create a temporary config file in current directory
        test_config = Path("sfunctor.yaml")
        test_config.touch()
        
        try:
            path = get_default_config_path()
            assert path == test_config
        finally:
            test_config.unlink()