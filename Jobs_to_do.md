# Jobs to run (reset for post-refactor reruns)

All prior runs need rerun with the unified histogram changes. Use `job_scripts/production/run_distributed_analysis_frontier_generic.sh` with 36h walltime.

## Node/parameter guide (per resolution)
- 640:   8 nodes,  Ndisp=8000,  Nrand=2000
- 1280: 16 nodes,  Ndisp=16000, Nrand=4000
- 2560: 32 nodes,  Ndisp=32000, Nrand=8000
- 5120: 64 nodes,  Ndisp=64000, Nrand=16000
- 10240: 64 nodes (cap), Ndisp=128000, Nrand=32000

## 5120 beta survey (sw = 2/3/5)
- [ ] Turb_5120_beta1_dedt025_plm
- [ ] Turb_5120_beta6_dedt025_plm
- [ ] Turb_5120_beta25_dedt025_plm
- [ ] Turb_5120_beta100_dedt025_plm

## beta = 25 resolution study (sw = 2/3/5; Nell=96)
- [ ] Turb_640_beta25_dedt025_plm
- [ ] Turb_1280_beta25_dedt025_plm
- [ ] Turb_2560_beta25_dedt025_plm
- [ ] Turb_5120_beta25_dedt025_plm
- [ ] Turb_10240_beta25_dedt025_plm

## Submission template
`sbatch -N <NODES> -t 36:00:00 --export=ALL,SIM_NAME=<SIM>,STENCIL_WIDTH=<SW>,N_DISP_TOTAL=<NDISP>,N_RANDOM_SUBSAMPLES=<NRAND>,N_ELL_BINS=96 job_scripts/production/run_distributed_analysis_frontier_generic.sh`
