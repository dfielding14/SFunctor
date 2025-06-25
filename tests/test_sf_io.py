"""Unit tests for sf_io module."""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from sfunctor.io.slice_io import parse_slice_metadata, load_slice_npz


class TestParseSliceMetadata:
    """Test cases for parse_slice_metadata function."""
    
    def test_valid_filename_patterns(self):
        """Test parsing of valid filename patterns."""
        test_cases = [
            ("slice_x1_-0.375_beta25_test.npz", (1, 25.0)),
            ("slice_x2_0.0_beta10.5_data.npz", (2, 10.5)),
            ("slice_x3_1.5_beta0.1_final.npz", (3, 0.1)),
            ("path/to/slice_x1_pos_beta100_file.npz", (1, 100.0)),
        ]
        
        for filename, expected in test_cases:
            axis, beta = parse_slice_metadata(filename)
            assert axis == expected[0]
            assert beta == expected[1]
    
    def test_invalid_input_types(self):
        """Test that invalid input types raise TypeError."""
        with pytest.raises(TypeError):
            parse_slice_metadata(123)
        
        with pytest.raises(TypeError):
            parse_slice_metadata(['not', 'a', 'string'])
        
        with pytest.raises(TypeError):
            parse_slice_metadata(None)
    
    def test_invalid_filename_patterns(self):
        """Test that invalid patterns raise ValueError."""
        invalid_patterns = [
            "not_a_slice_file.npz",
            "slice_x4_beta25.npz",  # x4 not valid
            "slice_y1_beta25.npz",  # y not x
            "slice_x1_nobeta.npz",  # missing beta
            "slice_x1_beta_notanumber.npz",  # beta not numeric
        ]
        
        for pattern in invalid_patterns:
            with pytest.raises(ValueError):
                parse_slice_metadata(pattern)
    
    def test_path_object_input(self):
        """Test that Path objects work correctly."""
        path = Path("slice_x2_0.0_beta5.5_test.npz")
        axis, beta = parse_slice_metadata(path)
        assert axis == 2
        assert beta == 5.5


class TestLoadSliceNpz:
    """Test cases for load_slice_npz function."""
    
    def setup_method(self):
        """Create temporary directory for test files."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)
    
    def create_test_npz(self, filename, data_dict):
        """Helper to create test .npz files."""
        file_path = self.temp_path / filename
        np.savez(file_path, **data_dict)
        return file_path
    
    def test_load_valid_file(self):
        """Test loading a valid slice file."""
        # Create test data
        shape = (100, 100)
        test_data = {
            'dens': np.ones(shape),
            'velx': np.full(shape, 2.0),
            'vely': np.full(shape, 3.0),
            'velz': np.full(shape, 4.0),
            'bcc1': np.full(shape, 0.1),
            'bcc2': np.full(shape, 0.2),
            'bcc3': np.full(shape, 0.3),
        }
        
        file_path = self.create_test_npz('test_slice.npz', test_data)
        result = load_slice_npz(file_path)
        
        # Check all expected keys are present
        expected_keys = {'rho', 'v_x', 'v_y', 'v_z', 'B_x', 'B_y', 'B_z',
                        'omega_x', 'omega_y', 'omega_z', 'j_x', 'j_y', 'j_z'}
        assert set(result.keys()) == expected_keys
        
        # Check loaded values
        assert np.allclose(result['rho'], 1.0)
        assert np.allclose(result['v_x'], 2.0)
        assert result['v_x'].shape == shape
    
    def test_load_with_stride(self):
        """Test loading with stride parameter."""
        shape = (100, 100)
        test_data = {'dens': np.arange(10000).reshape(shape)}
        
        file_path = self.create_test_npz('test_stride.npz', test_data)
        
        # Test different strides
        for stride in [1, 2, 5, 10]:
            result = load_slice_npz(file_path, stride=stride)
            expected_shape = (shape[0] // stride, shape[1] // stride)
            assert result['rho'].shape == expected_shape
    
    def test_missing_fields_filled_with_nan(self):
        """Test that missing fields are filled with NaN."""
        # Create file with only some fields
        test_data = {
            'dens': np.ones((50, 50)),
            'velx': np.ones((50, 50)),
        }
        
        file_path = self.create_test_npz('partial_data.npz', test_data)
        result = load_slice_npz(file_path)
        
        # Check that missing fields are NaN
        assert np.all(np.isnan(result['omega_x']))
        assert np.all(np.isnan(result['j_z']))
        
        # Check that present fields are loaded correctly
        assert np.allclose(result['rho'], 1.0)
    
    def test_invalid_stride(self):
        """Test that invalid stride values raise appropriate errors."""
        test_data = {'dens': np.ones((10, 10))}
        file_path = self.create_test_npz('test.npz', test_data)
        
        # Non-integer stride
        with pytest.raises(TypeError):
            load_slice_npz(file_path, stride=2.5)
        
        # Negative stride
        with pytest.raises(ValueError):
            load_slice_npz(file_path, stride=-1)
        
        # Zero stride
        with pytest.raises(ValueError):
            load_slice_npz(file_path, stride=0)
        
        # Stride too large
        with pytest.raises(ValueError):
            load_slice_npz(file_path, stride=20)
    
    def test_file_not_found(self):
        """Test that missing files raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_slice_npz('nonexistent_file.npz')
    
    def test_invalid_file_extension(self):
        """Test that non-.npz files raise ValueError."""
        test_file = self.temp_path / 'not_npz.txt'
        test_file.write_text('not an npz file')
        
        with pytest.raises(ValueError):
            load_slice_npz(test_file)
    
    def test_empty_file(self):
        """Test that empty .npz files raise ValueError."""
        file_path = self.create_test_npz('empty.npz', {})
        
        with pytest.raises(ValueError):
            load_slice_npz(file_path)
    
    def test_inconsistent_array_shapes(self):
        """Test that files with inconsistent array shapes raise ValueError."""
        test_data = {
            'dens': np.ones((100, 100)),
            'velx': np.ones((50, 50)),  # Different shape
        }
        
        file_path = self.create_test_npz('inconsistent.npz', test_data)
        
        with pytest.raises(ValueError):
            load_slice_npz(file_path)
    
    def test_non_2d_arrays(self):
        """Test that non-2D arrays raise ValueError."""
        test_data = {
            'dens': np.ones((100, 100, 3)),  # 3D array
        }
        
        file_path = self.create_test_npz('3d_data.npz', test_data)
        
        with pytest.raises(ValueError):
            load_slice_npz(file_path)