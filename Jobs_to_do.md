# Jobs to do (and done)

**Command template (5120)**: `sbatch -N 64 --export=ALL,SIM_NAME=<SIM>,STENCIL_WIDTH=<SW>,N_DISP_TOTAL=64000,N_RANDOM_SUBSAMPLES=16000,N_ELL_BINS=96 job_scripts/production/run_distributed_analysis_frontier_generic.sh`

## Completed (plots generated)
- [x] Turb_5120_beta6_dedt025_plm (sw2) — `/lustre/orion/ast207/proj-shared/dfielding/Production_plm/sfunctor_results/results_Turb_5120_beta6_dedt025_plm/ndisp64000_nrand16000_nell96_sw2_job3913352`
- [x] Turb_5120_beta25_dedt025_plm (sw2) — `/lustre/orion/ast207/proj-shared/dfielding/Production_plm/sfunctor_results/results_Turb_5120_beta25_dedt025_plm/ndisp64000_nrand16000_nell96_sw2_job3913414`
- [x] Turb_5120_beta100_dedt025_plm (sw2) — `/lustre/orion/ast207/proj-shared/dfielding/Production_plm/sfunctor_results/results_Turb_5120_beta100_dedt025_plm/ndisp64000_nrand16000_nell96_sw2_job3913448`

## Blocked
- [ ] Turb_5120_beta1_dedt025_plm (sw2) — plotting blocked: `sf_results_all_slices.npz` missing in `/lustre/orion/ast207/proj-shared/dfielding/Production_plm/sfunctor_results/results_Turb_5120_beta1_dedt025_plm/ndisp64000_nrand16000_nell96_sw2_job3913180`

## 5120 beta survey remaining
- sw2: [ ] beta1 (see Blocked), [ ] beta6 (done), [ ] beta25 (done), [ ] beta100 (done)
- sw3: [ ] beta1, [ ] beta6, [ ] beta25, [ ] beta100
- sw5: [ ] beta1, [ ] beta6, [ ] beta25, [ ] beta100

## beta = 25 resolution study (stencil widths 2/3/5; Nell=96)
N, Ndisp, Nrand by resolution: 640→(2, 8k, 2k); 1280→(16, 16k, 4k); 2560→(32, 32k, 8k); 5120→(64, 64k, 16k); 10240→(128, 128k, 32k).

- [ ] Turb_640_beta25_dedt025_plm (sw2/sw3/sw5)
- [ ] Turb_1280_beta25_dedt025_plm (sw2/sw3/sw5)
- [ ] Turb_2560_beta25_dedt025_plm (sw2/sw3/sw5)
- [ ] Turb_5120_beta25_dedt025_plm (sw2 done; sw3/sw5 pending)
- [ ] Turb_10240_beta25_dedt025_plm (sw2/sw3/sw5)

## Submission examples
- 5120 beta survey: use command template above; set `<SIM>=Turb_5120_beta<b>_dedt025_plm`, `<SW>=2|3|5`.
- beta25 resolution (sw2 example): `sbatch -N <N> --export=ALL,SIM_NAME=<SIM>,STENCIL_WIDTH=2,N_DISP_TOTAL=<Ndisp>,N_RANDOM_SUBSAMPLES=<Nrand>,N_ELL_BINS=96 job_scripts/production/run_distributed_analysis_frontier_generic.sh` (swap STENCIL_WIDTH to 3 or 5 as needed).
