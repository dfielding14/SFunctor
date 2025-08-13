# SFunctor Streamlining Implementation Plan

## Objective
Ruthlessly simplify SFunctor codebase for GPU optimization readiness by removing unnecessary complexity, following "boring is better" philosophy.

## Stage 1: Fix Critical Core Issues ✅ COMPLETE
**Goal**: Eliminate code duplication and fix broken APIs  
**Success Criteria**: Core functions work without patches, 50% less code in histograms  
**Tests**: Verify output matches current implementation exactly  
**Status**: COMPLETE

### Tasks:
- [x] Merge three stencil functions into one parameterized function
- [x] Simplify channel system to plain integer constants
- [x] Remove shared memory architecture entirely
- [x] Fix sf_bin_edges API inconsistency

## Stage 2: Remove Unnecessary Complexity ✅ COMPLETE
**Goal**: Strip out premature abstractions and over-engineering  
**Success Criteria**: No fake classes, single CLI pattern, minimal validation  
**Tests**: All existing functionality still works  
**Status**: COMPLETE

### Tasks:
- [x] Strip excessive validation from hot paths
- [x] Remove fake MPI classes - use simple conditionals
- [x] Consolidate CLI systems into one pattern
- [x] Standardize error handling approach

## Stage 3: Clean Architecture ✅ COMPLETE
**Goal**: Remove dead code and fix architectural issues  
**Success Criteria**: No duplicate scripts, clean imports, proper separation  
**Tests**: Package installs and runs cleanly  
**Status**: COMPLETE

### Tasks:
- [x] Remove dead/duplicate top-level scripts
- [x] Fix import patterns - no sys.path manipulation
- [x] Separate visualization config from IO modules
- [x] Clean up unused configuration options

## Stage 4: GPU Preparation ✅ COMPLETE
**Goal**: Create minimal, GPU-ready compute kernels  
**Success Criteria**: Core computation isolated from orchestration  
**Tests**: Performance benchmarks show improvement  
**Status**: COMPLETE

### Tasks:
- [x] Extract pure compute kernels without Python overhead
- [x] Make all loop bounds compile-time constants
- [x] Create chunked processing for large datasets
- [x] Validate all changes preserve correctness

## Design Principles Applied
- **Incremental progress**: Each stage independently testable
- **Boring solutions**: Remove clever code, add simple code
- **Single responsibility**: Each function does one thing
- **No premature abstraction**: Delete unused flexibility
- **Clear intent**: Function names describe what, not how

## Expected Outcomes
- 50% reduction in code size
- 5x faster GPU compilation potential
- 2x easier debugging
- 10x easier to extend for GPU

## Notes
- Maximum 3 attempts per issue before reassessing
- Every commit must compile and pass tests
- Document "why" in commit messages
- Stop and reassess if stuck