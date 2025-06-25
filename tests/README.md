# SFunctor Tests

This directory contains unit tests for the SFunctor package.

## Running Tests

### Quick Start

From the project root directory:

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=. --cov-report=term-missing

# Run specific test file
python -m pytest tests/test_sf_io.py

# Run with verbose output
python -m pytest -v
```

### Using the Test Runner

```bash
# From project root
python tests/run_tests.py
```

## Test Organization

- `test_sf_io.py` - Tests for I/O operations (loading slices, parsing metadata)
- `test_sf_physics.py` - Tests for physics calculations (Alfvén velocity, Elsasser variables)
- `test_sf_displacements.py` - Tests for displacement vector generation

## Writing New Tests

1. Create a new file `test_<module_name>.py` in the tests directory
2. Import pytest and the module to test
3. Group related tests in classes (optional)
4. Use descriptive test names starting with `test_`

Example:

```python
import pytest
import numpy as np
from my_module import my_function

class TestMyFunction:
    def test_basic_functionality(self):
        result = my_function(1, 2)
        assert result == 3
    
    def test_error_handling(self):
        with pytest.raises(ValueError):
            my_function(-1, 2)
```

## Test Coverage

Current test coverage focuses on:
- Input validation and error handling
- Edge cases (empty data, invalid values)
- Basic functionality verification
- Numerical accuracy checks

Areas for expansion:
- Integration tests for full pipeline
- Performance benchmarks
- MPI-specific tests
- Numba JIT compilation tests