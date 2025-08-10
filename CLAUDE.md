# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SFunctor is a high-performance Python pipeline for computing anisotropic, angle-resolved structure functions from 2D slices of 3D magnetohydrodynamic (MHD) simulations. It's designed for analyzing AthenaK simulation outputs with a focus on turbulence analysis.

## Common Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install numpy numba matplotlib cmasher mpi4py
```

### Running the Analysis
```bash
# Single slice analysis (no MPI)
python run_sf.py --file_name slice_data/slice_0000.npz --stride 2

# Multi-slice analysis with MPI
mpirun -n 64 python run_sf.py --slice_list slice_list.txt --stride 2

# Simplified analysis (for testing, no MPI/Numba)
python simple_sf_analysis.py --file_name slice_data/slice_0000.npz

# Extract 2D slice from 3D data
python extract_2d_slice.py --input_file data/Turb.hydro.00100.bin --output_file slice.npz --dimension y --slice_value 0

# Visualize results
python visualize_sf_results.py results_sf.npz
```

### Testing Approach
There's no formal test suite. For testing changes:
1. Use `demo_pipeline.py` for end-to-end workflow testing
2. Use `simple_sf_analysis.py` to test without MPI/Numba dependencies
3. Compare outputs between full and simplified versions
4. Test data available in `slice_data/` directory

## High-Level Architecture

### Core Pipeline Flow
1. **Data Input**: 2D slices from 3D MHD simulations (.npz format)
2. **Physics Computation**: Derive fields (vorticity, current, Alfvén variables)
3. **Displacement Generation**: Random vectors with configurable binning
4. **Structure Function Calculation**: 24-channel histograms via Numba kernels
5. **Output**: Self-describing .npz files with complete metadata

### Parallelization Strategy
- **Level 1**: MPI for multi-node distribution (embarrassingly parallel across slices)
- **Level 2**: Shared-memory multiprocessing within nodes
- Graceful fallback when MPI not available

### Key Modules and Their Roles
- `run_sf.py`: Main entry point with MPI support
- `sf_cli.py`: Configuration management and argument parsing
- `sf_physics.py`: Physics calculations (must maintain consistency with AthenaK)
- `sf_histograms.py`: Performance-critical Numba kernels (24 channels)
- `sf_parallel.py`: Shared-memory parallelization logic
- `sf_displacements.py`: Displacement vector generation and binning

### Performance Considerations
- Numba JIT compilation for histogram kernels (first run slower)
- Minimal imports in worker processes to reduce overhead
- Configurable stencil widths (2, 3, or 5 points) affect memory/accuracy trade-off
- Memory usage scales with number of displacement vectors and bins

### Physics Channels Computed
The pipeline computes 24 structure function channels including:
- Velocity, magnetic field, density increments
- Elsasser variables (z+ and z-)
- Vorticity and current density
- Various cross products and angle-resolved statistics

When modifying physics calculations, ensure consistency with the channel definitions in `sf_histograms.py`.

### notes for what to do next:
1. On the New Machine

Clone or Pull the Repository

# If starting fresh on the new machine
git clone <your-repository-url>
cd SFunctor

# OR if you already have the repo there
cd SFunctor
git pull origin main

Set Up the Environment

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install the package in development mode
pip install -e .

# OR just install requirements
pip install -r requirements.txt

3. Key Information to Share with Claude on the New Machine

When you start the session on the new machine, share this context:

1. Current State: All major improvements have been completed except memory
optimizations:
  - ✅ Package restructured into proper Python package
  - ✅ Comprehensive error handling added
  - ✅ Unit tests created
  - ✅ Documentation improved
  - ✅ Configuration file support added
  - ⏳ Memory optimizations pending (requires large dataset)
2. Memory Optimization Focus Areas:
  - The compute_histograms_shared function in sfunctor/core/parallel.py creates
 shared memory for all field arrays
  - The histogram accumulation in sfunctor/core/histograms.py might benefit
from chunking
  - Consider streaming/chunking large slice files instead of loading entirely
into memory
  - Profile memory usage during displacement vector processing
3. Testing Memory Optimizations:
  - Use a configuration profile with reduced settings first:
  python run_analysis.py --config examples/configs/profiles.yaml --profile
lowmem --file_name <large_slice.npz>
  - Monitor memory usage with tools like htop or memory profiler
  - The current implementation loads all fields into shared memory which might
be problematic for very large slices

4. Files That Might Need Memory Optimization

1. sfunctor/io/slice_io.py - Currently loads entire slice into memory
2. sfunctor/core/parallel.py - Creates shared memory copies of all fields
3. sfunctor/core/histograms.py - Processes all displacements in memory
4. sfunctor/analysis/batch.py - Might need chunking for very large slices

5. Testing Commands for Large Data

# Test current memory usage
python -m memory_profiler run_analysis.py --config sfunctor.yaml --profile
lowmem --file_name <large_slice.npz>

# Use reduced sampling for initial tests
python run_analysis.py --stride 8 --n_disp_total 1000 --N_random_subsamples 500
 --file_name <large_slice.npz>

6. What to Look For

- Peak memory usage during:
  - Slice loading
  - Shared memory creation
  - Histogram computation
- Whether memory scales with:
  - Slice size (N×N grid)
  - Number of displacements
  - Number of processes

The code is now well-structured for optimization work, with clear module
boundaries and good error handling that will help identify any memory-related
issues.

# Development Guidelines

## Philosophy

### Core Beliefs

- **Incremental progress over big bangs** - Small changes that compile and pass tests
- **Learning from existing code** - Study and plan before implementing
- **Pragmatic over dogmatic** - Adapt to project reality
- **Clear intent over clever code** - Be boring and obvious

### Simplicity Means

- Single responsibility per function/class
- Avoid premature abstractions
- No clever tricks - choose the boring solution
- If you need to explain it, it's too complex

## Process

### 1. Planning & Staging

Break complex work into 3-5 stages. Document in `IMPLEMENTATION_PLAN.md`:

```markdown
## Stage N: [Name]
**Goal**: [Specific deliverable]
**Success Criteria**: [Testable outcomes]
**Tests**: [Specific test cases]
**Status**: [Not Started|In Progress|Complete]
```
- Update status as you progress
- Remove file when all stages are done

### 2. Implementation Flow

1. **Understand** - Study existing patterns in codebase
2. **Test** - Write test first (red)
3. **Implement** - Minimal code to pass (green)
4. **Refactor** - Clean up with tests passing
5. **Commit** - With clear message linking to plan

### 3. When Stuck (After 3 Attempts)

**CRITICAL**: Maximum 3 attempts per issue, then STOP.

1. **Document what failed**:
   - What you tried
   - Specific error messages
   - Why you think it failed

2. **Research alternatives**:
   - Find 2-3 similar implementations
   - Note different approaches used

3. **Question fundamentals**:
   - Is this the right abstraction level?
   - Can this be split into smaller problems?
   - Is there a simpler approach entirely?

4. **Try different angle**:
   - Different library/framework feature?
   - Different architectural pattern?
   - Remove abstraction instead of adding?

## Technical Standards

### Architecture Principles

- **Composition over inheritance** - Use dependency injection
- **Interfaces over singletons** - Enable testing and flexibility
- **Explicit over implicit** - Clear data flow and dependencies
- **Test-driven when possible** - Never disable tests, fix them

### Code Quality

- **Every commit must**:
  - Compile successfully
  - Pass all existing tests
  - Include tests for new functionality
  - Follow project formatting/linting

- **Before committing**:
  - Run formatters/linters
  - Self-review changes
  - Ensure commit message explains "why"

### Error Handling

- Fail fast with descriptive messages
- Include context for debugging
- Handle errors at appropriate level
- Never silently swallow exceptions

## Decision Framework

When multiple valid approaches exist, choose based on:

1. **Testability** - Can I easily test this?
2. **Readability** - Will someone understand this in 6 months?
3. **Consistency** - Does this match project patterns?
4. **Simplicity** - Is this the simplest solution that works?
5. **Reversibility** - How hard to change later?

## Project Integration

### Learning the Codebase

- Find 3 similar features/components
- Identify common patterns and conventions
- Use same libraries/utilities when possible
- Follow existing test patterns

### Tooling

- Use project's existing build system
- Use project's test framework
- Use project's formatter/linter settings
- Don't introduce new tools without strong justification

## Quality Gates

### Definition of Done

- [ ] Tests written and passing
- [ ] Code follows project conventions
- [ ] No linter/formatter warnings
- [ ] Commit messages are clear
- [ ] Implementation matches plan
- [ ] No TODOs without issue numbers

### Test Guidelines

- Test behavior, not implementation
- One assertion per test when possible
- Clear test names describing scenario
- Use existing test utilities/helpers
- Tests should be deterministic

## Important Reminders

**NEVER**:
- Use `--no-verify` to bypass commit hooks
- Disable tests instead of fixing them
- Commit code that doesn't compile
- Make assumptions - verify with existing code

**ALWAYS**:
- Commit working code incrementally
- Update plan documentation as you go
- Learn from existing implementations
- Stop after 3 failed attempts and reassess

