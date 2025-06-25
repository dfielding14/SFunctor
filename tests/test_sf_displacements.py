"""Unit tests for sf_displacements module."""

import pytest
import numpy as np
import warnings

from sfunctor.utils.displacements import find_ell_bin_edges, build_displacement_list


class TestFindEllBinEdges:
    """Test cases for find_ell_bin_edges function."""
    
    def test_basic_functionality(self):
        """Test basic bin edge generation."""
        edges = find_ell_bin_edges(1.0, 100.0, 10)
        
        # Check we get correct number of edges (or close)
        assert len(edges) >= 10  # May get 11 or close
        assert len(edges) <= 12  # Should not exceed by much
        
        # Check edges are monotonic
        assert np.all(np.diff(edges) > 0)
        
        # Check range
        assert edges[0] >= 1.0
        assert edges[-1] <= 100.0
        
        # Check they are integers (as floats)
        assert np.allclose(edges, np.round(edges))
    
    def test_small_range(self):
        """Test with small range of values."""
        edges = find_ell_bin_edges(1.0, 10.0, 5)
        
        # With small range, might not get exactly requested bins
        assert len(edges) >= 3
        assert len(edges) <= 7
        
        # Check edges are unique integers
        assert len(np.unique(edges)) == len(edges)
    
    def test_large_number_of_bins(self):
        """Test with large number of requested bins."""
        edges = find_ell_bin_edges(1.0, 1000.0, 50)
        
        # Should get close to requested number
        assert len(edges) >= 40
        assert len(edges) <= 60
        
        # Check logarithmic spacing (approximately)
        log_edges = np.log10(edges)
        diffs = np.diff(log_edges)
        assert np.std(diffs) < 0.5  # Roughly uniform in log space
    
    def test_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        # Negative r_min
        with pytest.raises(ValueError):
            find_ell_bin_edges(-1.0, 100.0, 10)
        
        # Zero r_min
        with pytest.raises(ValueError):
            find_ell_bin_edges(0.0, 100.0, 10)
        
        # r_max <= r_min
        with pytest.raises(ValueError):
            find_ell_bin_edges(100.0, 50.0, 10)
        
        with pytest.raises(ValueError):
            find_ell_bin_edges(100.0, 100.0, 10)
        
        # Invalid n_ell_bins
        with pytest.raises(ValueError):
            find_ell_bin_edges(1.0, 100.0, 0)
        
        with pytest.raises(ValueError):
            find_ell_bin_edges(1.0, 100.0, -5)
    
    def test_warning_for_impossible_request(self):
        """Test warning when exact number of bins cannot be achieved."""
        # Very small range with many bins requested
        with warnings.catch_warnings(record=True) as w:
            edges = find_ell_bin_edges(1.0, 3.0, 20)
            
            # Should warn about not achieving exact number
            assert len(w) >= 1
            assert "could not achieve exactly" in str(w[0].message).lower()
        
        # Should still return something reasonable
        assert len(edges) >= 2
        assert len(edges) <= 5  # Can't have many unique integers between 1 and 3
    
    def test_edge_cases(self):
        """Test edge cases."""
        # Single bin
        edges = find_ell_bin_edges(1.0, 100.0, 1)
        assert len(edges) == 2
        assert edges[0] >= 1.0
        assert edges[-1] <= 100.0
        
        # Very close r_min and r_max
        edges = find_ell_bin_edges(10.0, 11.0, 1)
        assert len(edges) == 2
        assert edges[0] == 10.0
        assert edges[1] == 11.0


class TestBuildDisplacementList:
    """Test cases for build_displacement_list function."""
    
    def test_basic_functionality(self):
        """Test basic displacement generation."""
        edges = np.array([1.0, 10.0, 100.0])
        disps = build_displacement_list(edges, 1000, seed=42)
        
        # Check shape
        assert disps.ndim == 2
        assert disps.shape[1] == 2
        
        # Check we get reasonable number of displacements
        # (less than requested due to de-duplication)
        assert len(disps) > 100
        assert len(disps) < 1000
        
        # Check all displacements are integers
        assert disps.dtype in [np.int32, np.int64]
        
        # Check displacements are sorted by magnitude
        mags = np.sqrt(disps[:, 0]**2 + disps[:, 1]**2)
        assert np.all(np.diff(mags) >= 0)
    
    def test_displacement_range(self):
        """Test that displacements fall within expected range."""
        edges = np.array([5.0, 20.0, 50.0])
        disps = build_displacement_list(edges, 500, seed=42)
        
        mags = np.sqrt(disps[:, 0]**2 + disps[:, 1]**2)
        
        # Check range (with some tolerance for rounding)
        assert np.min(mags) >= 4.0
        assert np.max(mags) <= 51.0
    
    def test_no_zero_displacement(self):
        """Test that (0, 0) displacement is not included."""
        edges = np.array([1.0, 10.0])
        disps = build_displacement_list(edges, 100, seed=42)
        
        # Check no zero displacement
        assert not np.any((disps[:, 0] == 0) & (disps[:, 1] == 0))
    
    def test_mirror_removal(self):
        """Test that mirror duplicates are removed."""
        edges = np.array([1.0, 10.0])
        disps = build_displacement_list(edges, 200, seed=42)
        
        # For each displacement, check its negative is not present
        for dx, dy in disps:
            if dx != 0 or dy != 0:  # Skip if we somehow got (0,0)
                neg_present = np.any((disps[:, 0] == -dx) & (disps[:, 1] == -dy))
                assert not neg_present, f"Found both ({dx},{dy}) and ({-dx},{-dy})"
    
    def test_reproducibility_with_seed(self):
        """Test that same seed gives same results."""
        edges = np.array([1.0, 50.0])
        
        disps1 = build_displacement_list(edges, 500, seed=123)
        disps2 = build_displacement_list(edges, 500, seed=123)
        
        assert np.array_equal(disps1, disps2)
    
    def test_different_seeds(self):
        """Test that different seeds give different results."""
        edges = np.array([1.0, 50.0])
        
        disps1 = build_displacement_list(edges, 500, seed=123)
        disps2 = build_displacement_list(edges, 500, seed=456)
        
        # Should be different (very unlikely to be identical)
        assert not np.array_equal(disps1, disps2)
    
    def test_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        # Non-array edges
        with pytest.raises(TypeError):
            build_displacement_list([1, 10, 100], 100)
        
        # Too few edges
        with pytest.raises(ValueError):
            build_displacement_list(np.array([10.0]), 100)
        
        # Non-monotonic edges
        with pytest.raises(ValueError):
            build_displacement_list(np.array([1.0, 50.0, 30.0]), 100)
        
        # Invalid n_disp_total
        with pytest.raises(ValueError):
            build_displacement_list(np.array([1.0, 10.0]), 0)
        
        with pytest.raises(ValueError):
            build_displacement_list(np.array([1.0, 10.0]), -10)
    
    def test_warning_for_few_displacements(self):
        """Test warning when too few displacements per bin."""
        edges = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # 5 bins
        
        with warnings.catch_warnings(record=True) as w:
            # Only 10 total for 5 bins = 2 per bin
            disps = build_displacement_list(edges, 10)
            
            assert len(w) >= 1
            assert "consider increasing" in str(w[0].message).lower()
    
    def test_empty_bins_handling(self):
        """Test handling when bins might be empty."""
        # Very narrow bins
        edges = np.array([10.0, 10.5, 11.0])
        disps = build_displacement_list(edges, 20, seed=42)
        
        # Should still get some displacements
        assert len(disps) > 0
        
        # All should be around magnitude 10-11
        mags = np.sqrt(disps[:, 0]**2 + disps[:, 1]**2)
        assert np.all(mags >= 9)
        assert np.all(mags <= 12)
    
    def test_large_displacement_request(self):
        """Test with large number of requested displacements."""
        edges = np.array([1.0, 10.0, 100.0, 1000.0])
        disps = build_displacement_list(edges, 10000, seed=42)
        
        # Should get many unique displacements
        assert len(disps) > 1000
        
        # Check distribution across scales
        mags = np.sqrt(disps[:, 0]**2 + disps[:, 1]**2)
        
        # Should have displacements at all scales
        assert np.sum(mags < 10) > 0  # Small scale
        assert np.sum((mags >= 10) & (mags < 100)) > 0  # Medium scale
        assert np.sum(mags >= 100) > 0  # Large scale