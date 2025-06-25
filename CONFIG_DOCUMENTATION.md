# Configuration File Support

SFunctor now supports YAML configuration files to simplify parameter management and improve reproducibility.

## Quick Start

1. **Create a configuration file:**
   ```bash
   python create_config.py
   ```
   This creates `sfunctor.yaml` with all available parameters.

2. **Use the configuration:**
   ```bash
   python run_analysis.py --config sfunctor.yaml --file_name slice.npz
   ```

3. **Use a profile:**
   ```bash
   python run_analysis.py --config sfunctor.yaml --profile production --file_name slice.npz
   ```

## Configuration File Structure

Configuration files use YAML format with these main sections:

### Analysis Parameters
```yaml
analysis:
  stride: 2                  # Spatial downsampling
  n_disp_total: 10000       # Number of displacement vectors
  N_random_subsamples: 5000 # Random samples per displacement
  n_ell_bins: 128           # Displacement magnitude bins
  stencil_width: 2          # Finite difference stencil (2, 3, or 5)
  n_processes: 0            # Worker processes (0 = auto)
```

### Binning Configuration
```yaml
binning:
  n_theta_bins: 18          # Angular bins (theta)
  n_phi_bins: 18            # Angular bins (phi)
  sf_min: 1.0e-4           # Min structure function value
  sf_max: 10.0             # Max structure function value
  sf_bins: 128             # Structure function bins
  product_min: 1.0e-5      # Min cross-product value
  product_max: 1.0e5       # Max cross-product value
  product_bins: 128        # Cross-product bins
```

### Output Settings
```yaml
output:
  format: "npz"            # Output format
  compression: true        # Enable compression
  output_dir: "./results"  # Output directory
  save_intermediate: false # Save intermediate results
```

### Physics Parameters
```yaml
physics:
  min_density: 1.0e-10     # Minimum density threshold
```

## Profiles

Profiles allow you to define multiple configurations in one file:

```yaml
profiles:
  quick_test:
    analysis:
      n_disp_total: 100
      N_random_subsamples: 100
      n_ell_bins: 32
      stride: 8
  
  production:
    analysis:
      stride: 1
      n_disp_total: 20000
      N_random_subsamples: 10000
      stencil_width: 3
```

Use a profile with: `--profile production`

## Configuration Precedence

Parameters are applied in this order (later overrides earlier):
1. Built-in defaults
2. Configuration file values
3. Profile values (if specified)
4. Command-line arguments

Example:
```bash
# Config file sets stride: 2
# Command line overrides to stride: 4
python run_analysis.py --config sfunctor.yaml --stride 4 --file_name slice.npz
```

## Default Configuration Locations

SFunctor looks for configuration files in this order:
1. `sfunctor.yaml` (current directory)
2. `.sfunctor.yaml` (current directory) 
3. `~/.sfunctor/config.yaml` (home directory)
4. Path in `$SFUNCTOR_CONFIG` environment variable
5. Path specified with `--config`

## Example Configuration Files

See the `examples/configs/` directory for:
- `default.yaml` - Standard analysis settings
- `profiles.yaml` - Multiple profiles for different use cases
- `beta_study.yaml` - Optimized for plasma beta parameter studies

## Command-Line Options

New configuration-related options:
- `--config PATH` - Specify configuration file
- `--profile NAME` - Use a named profile from the config
- `--print-config` - Show effective configuration and exit

## Tips

1. **Version Control**: Check your config files into git for reproducibility
2. **Comments**: YAML supports comments - document your parameter choices
3. **Validation**: Invalid parameters are caught early with helpful messages
4. **Overrides**: Command-line arguments always win over config file values
5. **Templates**: Use `create_config.py` to generate a fully-commented template

## Python API

```python
from sfunctor.utils import load_config, create_config_template

# Load a config file
config = load_config("my_config.yaml", profile="production")

# Create a template
template = create_config_template()
with open("new_config.yaml", "w") as f:
    f.write(template)
```