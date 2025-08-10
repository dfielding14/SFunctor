# Implementation Plan: Fix Shared Memory Implementation

## Problem Statement

The current `run_node_analysis.py` uses Python's standard `multiprocessing.Pool` which causes memory duplication when processing large 10240×10240 slices. With 56 processes and 18 GB of field data, this leads to out-of-memory (OOM) errors. The sfunctor package already has a proper shared memory implementation in `sfunctor/core/parallel.py` that should be used instead.

## Root Cause Analysis

1. **Current Implementation Issue**: `run_node_analysis.py` passes the entire `fields` dictionary to each process via pickle serialization (line 199-200), causing each of 56 processes to have its own copy of the 18 GB data.

2. **Existing Solution**: The `compute_histograms_shared` function in `sfunctor/core/parallel.py` properly:
   - Creates shared memory segments once
   - Workers attach to the same memory without duplication
   - Marks arrays as read-only to prevent corruption
   - Handles cleanup automatically

## Success Criteria

- [ ] Jobs complete successfully with 56 processes on 10240×10240 slices
- [ ] Memory usage stays below 100 GB per node (currently exceeds 512 GB)
- [ ] No change in computation results (histogram values remain identical)
- [ ] All existing tests pass
- [ ] Performance improves or remains the same

---

## Stage 1: Study and Validate Existing Implementation
**Goal**: Understand the existing shared memory implementation and create validation tests  
**Success Criteria**: 
- Document how `compute_histograms_shared` works
- Create a test that verifies shared memory is actually shared
- Confirm API compatibility with current usage

**Tests**: 
- Unit test comparing results between Pool and shared memory implementations
- Memory usage test showing no duplication with shared memory

**Status**: Complete ✓

### Tasks:
1. ✓ Studied `sfunctor/core/parallel.py` implementation in detail
2. ✓ Studied current `run_node_analysis.py` usage patterns
3. ✓ Created test files to validate behavior
4. ✓ Documented the API differences and requirements

### Results:
- Confirmed compute_histograms_shared creates shared memory segments once
- Verified workers attach without duplication
- API requires passing all displacements at once (no manual batching)

---

## Stage 2: Create Minimal Integration
**Goal**: Replace Pool implementation with shared memory in a minimal test case  
**Success Criteria**: 
- Test script processes a small slice (512×512) successfully
- Memory usage is demonstrably lower
- Results match exactly

**Tests**: 
- Integration test with small test data
- Memory profiling comparison

**Status**: Complete ✓

### Tasks:
1. ✓ Created `run_node_analysis_shared.py` as minimal integration example
2. ✓ Created test scripts to verify identical results
3. ✓ Added memory usage reporting to implementation
4. ✓ Documented API adjustments (no batching, single sf_bin_edges)

---

## Stage 3: Refactor run_node_analysis.py
**Goal**: Replace the Pool implementation with compute_histograms_shared  
**Success Criteria**: 
- `run_node_analysis.py` uses shared memory implementation
- Backward compatibility maintained (same CLI interface)
- No regression in functionality

**Tests**: 
- Existing test_node_analysis.sh still works
- Full integration test with actual data

**Status**: Complete ✓

### Tasks:
1. ✓ Original backed up in git history
2. ✓ Imported and integrated `compute_histograms_shared`
3. ✓ Removed process_displacement_batch function and batching logic
4. ✓ Maintained all metadata and output format
5. ✓ Added memory usage reporting for monitoring

---

## Stage 4: Validate on Production Data
**Goal**: Ensure the fix works with full-scale production data  
**Success Criteria**: 
- Process 10240×10240 slices successfully
- Complete a full job with 64 nodes
- Memory usage stays within limits

**Tests**: 
- Run test_node_analysis.sh with 10240×10240 slice
- Submit test job with single node
- Submit full production job

**Status**: In Progress ⏳

### Tasks:
1. ✓ Created test_production_validation.sh for testing
2. ⏳ Test with single node, single slice (ready to run)
3. ⏳ Test with single node, multiple slices  
4. ⏳ Test with multiple nodes (small job)
5. ⏳ Run full production job
6. ⏳ Compare results with previous runs

---

## Stage 5: Documentation and Cleanup
**Goal**: Document changes and clean up  
**Success Criteria**: 
- Code is well-documented
- CLAUDE.md updated with memory optimization notes
- Old code removed or deprecated

**Tests**: 
- Documentation review
- Final regression test suite

**Status**: Not Started

### Tasks:
1. Add docstrings explaining shared memory usage
2. Update CLAUDE.md with lessons learned
3. Remove or deprecate old Pool-based code
4. Create troubleshooting guide for memory issues
5. Delete this IMPLEMENTATION_PLAN.md

---

## Risk Mitigation

### Potential Issues and Mitigation Strategies:

1. **API Incompatibility**
   - Risk: `compute_histograms_shared` might expect different data format
   - Mitigation: Create adapter layer if needed
   - Fallback: Keep Pool implementation as option via flag

2. **Performance Regression**
   - Risk: Shared memory might be slower than expected
   - Mitigation: Profile and optimize hot paths
   - Fallback: Tune number of processes

3. **Platform-Specific Issues**
   - Risk: Shared memory might behave differently on Frontier
   - Mitigation: Test on compute nodes early
   - Fallback: Use reduced process count (14 instead of 56)

## Notes

- Current workaround (14 processes) works but is 4x slower
- Priority is correctness over speed
- Must maintain backward compatibility
- Follow incremental development approach per CLAUDE.md guidelines