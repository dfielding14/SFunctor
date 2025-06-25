"""Unit tests for sf_physics module."""

import pytest
import numpy as np
import warnings

from sfunctor.core.physics import compute_vA, compute_z_plus_minus


class TestComputeVA:
    """Test cases for compute_vA function."""
    
    def test_basic_computation(self):
        """Test basic Alfvén velocity computation."""
        # Simple case: uniform fields
        shape = (10, 10)
        B_x = np.ones(shape)
        B_y = np.zeros(shape)
        B_z = np.zeros(shape)
        rho = np.full(shape, 4.0)
        
        vA_x, vA_y, vA_z = compute_vA(B_x, B_y, B_z, rho)
        
        # v_A = B / sqrt(rho) = 1 / sqrt(4) = 0.5
        assert np.allclose(vA_x, 0.5)
        assert np.allclose(vA_y, 0.0)
        assert np.allclose(vA_z, 0.0)
    
    def test_varying_fields(self):
        """Test with spatially varying fields."""
        shape = (5, 5)
        B_x = np.arange(25).reshape(shape)
        B_y = np.arange(25).reshape(shape) * 2
        B_z = np.arange(25).reshape(shape) * 3
        rho = np.full(shape, 9.0)
        
        vA_x, vA_y, vA_z = compute_vA(B_x, B_y, B_z, rho)
        
        # Check shapes
        assert vA_x.shape == shape
        assert vA_y.shape == shape
        assert vA_z.shape == shape
        
        # Check specific values
        assert np.allclose(vA_x, B_x / 3.0)
        assert np.allclose(vA_y, B_y / 3.0)
        assert np.allclose(vA_z, B_z / 3.0)
    
    def test_invalid_input_types(self):
        """Test that non-numpy arrays raise TypeError."""
        shape = (5, 5)
        B_x = [[1] * 5] * 5  # List, not numpy array
        B_y = np.zeros(shape)
        B_z = np.zeros(shape)
        rho = np.ones(shape)
        
        with pytest.raises(TypeError):
            compute_vA(B_x, B_y, B_z, rho)
    
    def test_invalid_dimensions(self):
        """Test that non-2D arrays raise ValueError."""
        # 1D arrays
        with pytest.raises(ValueError):
            compute_vA(np.ones(10), np.ones(10), np.ones(10), np.ones(10))
        
        # 3D arrays
        shape_3d = (5, 5, 5)
        with pytest.raises(ValueError):
            compute_vA(np.ones(shape_3d), np.ones(shape_3d), 
                      np.ones(shape_3d), np.ones(shape_3d))
    
    def test_shape_mismatch(self):
        """Test that mismatched shapes raise ValueError."""
        B_x = np.ones((10, 10))
        B_y = np.ones((10, 10))
        B_z = np.ones((10, 10))
        rho = np.ones((5, 5))  # Different shape
        
        with pytest.raises(ValueError):
            compute_vA(B_x, B_y, B_z, rho)
    
    def test_negative_density(self):
        """Test handling of negative density values."""
        shape = (5, 5)
        B_x = np.ones(shape)
        B_y = np.zeros(shape)
        B_z = np.zeros(shape)
        rho = np.full(shape, 1.0)
        rho[2, 2] = -1.0  # One negative value
        
        with warnings.catch_warnings(record=True) as w:
            vA_x, vA_y, vA_z = compute_vA(B_x, B_y, B_z, rho)
            
            # Check that warning was issued
            assert len(w) == 1
            assert "non-positive density" in str(w[0].message).lower()
        
        # Check that Alfvén speed is zero where density was negative
        assert vA_x[2, 2] == 0.0
        assert vA_y[2, 2] == 0.0
        assert vA_z[2, 2] == 0.0
        
        # Check other values are computed correctly
        assert np.allclose(vA_x[0, 0], 1.0)
    
    def test_zero_density(self):
        """Test handling of zero density values."""
        shape = (3, 3)
        B_x = np.ones(shape)
        B_y = np.ones(shape)
        B_z = np.ones(shape)
        rho = np.ones(shape)
        rho[1, 1] = 0.0
        
        with warnings.catch_warnings(record=True):
            vA_x, vA_y, vA_z = compute_vA(B_x, B_y, B_z, rho)
        
        # Alfvén speed should be zero where density is zero
        assert vA_x[1, 1] == 0.0
        assert vA_y[1, 1] == 0.0
        assert vA_z[1, 1] == 0.0


class TestComputeZPlusMinus:
    """Test cases for compute_z_plus_minus function."""
    
    def test_basic_computation(self):
        """Test basic Elsasser variable computation."""
        shape = (10, 10)
        v_x = np.full(shape, 3.0)
        v_y = np.full(shape, 4.0)
        v_z = np.full(shape, 5.0)
        vA_x = np.full(shape, 1.0)
        vA_y = np.full(shape, 2.0)
        vA_z = np.full(shape, 3.0)
        
        z_plus, z_minus = compute_z_plus_minus(v_x, v_y, v_z, vA_x, vA_y, vA_z)
        
        # z+ = v + vA
        assert np.allclose(z_plus[0], 4.0)  # 3 + 1
        assert np.allclose(z_plus[1], 6.0)  # 4 + 2
        assert np.allclose(z_plus[2], 8.0)  # 5 + 3
        
        # z- = v - vA
        assert np.allclose(z_minus[0], 2.0)  # 3 - 1
        assert np.allclose(z_minus[1], 2.0)  # 4 - 2
        assert np.allclose(z_minus[2], 2.0)  # 5 - 3
    
    def test_zero_fields(self):
        """Test with zero velocity or Alfvén speed."""
        shape = (5, 5)
        v_x = np.zeros(shape)
        v_y = np.zeros(shape)
        v_z = np.zeros(shape)
        vA_x = np.ones(shape)
        vA_y = np.ones(shape)
        vA_z = np.ones(shape)
        
        z_plus, z_minus = compute_z_plus_minus(v_x, v_y, v_z, vA_x, vA_y, vA_z)
        
        # z+ = 0 + 1 = 1
        assert np.allclose(z_plus[0], 1.0)
        assert np.allclose(z_plus[1], 1.0)
        assert np.allclose(z_plus[2], 1.0)
        
        # z- = 0 - 1 = -1
        assert np.allclose(z_minus[0], -1.0)
        assert np.allclose(z_minus[1], -1.0)
        assert np.allclose(z_minus[2], -1.0)
    
    def test_shape_consistency(self):
        """Test that output shapes match input shapes."""
        shape = (7, 9)
        v_x = np.random.rand(*shape)
        v_y = np.random.rand(*shape)
        v_z = np.random.rand(*shape)
        vA_x = np.random.rand(*shape)
        vA_y = np.random.rand(*shape)
        vA_z = np.random.rand(*shape)
        
        z_plus, z_minus = compute_z_plus_minus(v_x, v_y, v_z, vA_x, vA_y, vA_z)
        
        # Check shapes
        assert z_plus[0].shape == shape
        assert z_plus[1].shape == shape
        assert z_plus[2].shape == shape
        assert z_minus[0].shape == shape
        assert z_minus[1].shape == shape
        assert z_minus[2].shape == shape
    
    def test_invalid_input_types(self):
        """Test that non-numpy arrays raise TypeError."""
        shape = (5, 5)
        v_x = [[1] * 5] * 5  # List, not numpy array
        v_y = np.zeros(shape)
        v_z = np.zeros(shape)
        vA_x = np.zeros(shape)
        vA_y = np.zeros(shape)
        vA_z = np.zeros(shape)
        
        with pytest.raises(TypeError):
            compute_z_plus_minus(v_x, v_y, v_z, vA_x, vA_y, vA_z)
    
    def test_shape_mismatch(self):
        """Test that mismatched shapes raise ValueError."""
        v_x = np.ones((10, 10))
        v_y = np.ones((10, 10))
        v_z = np.ones((10, 10))
        vA_x = np.ones((5, 5))  # Different shape
        vA_y = np.ones((10, 10))
        vA_z = np.ones((10, 10))
        
        with pytest.raises(ValueError):
            compute_z_plus_minus(v_x, v_y, v_z, vA_x, vA_y, vA_z)
    
    def test_nan_handling(self):
        """Test behavior with NaN values."""
        shape = (3, 3)
        v_x = np.ones(shape)
        v_y = np.ones(shape)
        v_z = np.ones(shape)
        vA_x = np.ones(shape)
        vA_y = np.ones(shape)
        vA_z = np.ones(shape)
        
        # Introduce NaN
        v_x[1, 1] = np.nan
        
        with warnings.catch_warnings(record=True) as w:
            z_plus, z_minus = compute_z_plus_minus(v_x, v_y, v_z, vA_x, vA_y, vA_z)
            
            # Check that warning was issued
            assert len(w) == 1
            assert "non-finite" in str(w[0].message).lower()
        
        # Check that NaN propagates
        assert np.isnan(z_plus[0][1, 1])
        assert np.isnan(z_minus[0][1, 1])
    
    def test_infinite_values(self):
        """Test behavior with infinite values."""
        shape = (3, 3)
        v_x = np.ones(shape)
        v_y = np.ones(shape)
        v_z = np.ones(shape)
        vA_x = np.ones(shape)
        vA_y = np.ones(shape)
        vA_z = np.ones(shape)
        
        # Introduce infinity
        vA_y[0, 0] = np.inf
        
        with warnings.catch_warnings(record=True) as w:
            z_plus, z_minus = compute_z_plus_minus(v_x, v_y, v_z, vA_x, vA_y, vA_z)
            
            # Check that warning was issued
            assert len(w) == 1
            assert "non-finite" in str(w[0].message).lower()
        
        # Check that infinity propagates correctly
        assert z_plus[1][0, 0] == np.inf
        assert z_minus[1][0, 0] == -np.inf