#!/bin/bash
# N2-E3-S15 (plan.md): xác nhận σ = 1,5 trên IN100 + IN100-C, rồi chấm điểm.
cd /home/tbm/Documents/Input_SIMP_Analyst
PY=/home/tbm/miniconda3/envs/simp/bin/python
B="$PY -u pipeline/phase5_cvae/benchmark_physics_guided_refinement.py --cvae-ckpt outputs/phase5/cvae_v2_finetuned.pt --n-conditions 100 --n-samples 30 --steps 32 --projection-betas 1 4 16 64 --n-boot 10000 --save-images --design-filter-sigma 1.5"
O=outputs/phase5/plan_v3/n2
( $B --seed 123 --out $O/e3s15_in100.json > $O/e3s15_in100.log 2>&1; echo "IN100 EXIT $?" >> $O/e3s15_in100.log ) &
( $B --seed 789 --out $O/e3s15_in100c.json > $O/e3s15_in100c.log 2>&1; echo "IN100C EXIT $?" >> $O/e3s15_in100c.log ) &
wait
cd $O
echo "== chấm IN100 $(date +%T)"
$PY -u eval_general.py --kinds guarded --cache cache_in100.json --out e3s15_scores_in100.json F=e3f_in100_filter.json F15=e3s15_in100.json A=../p1_7_c5_in100_bo30_nofp.json SIMP=../p1_6a_simp_baseline_in100.json
echo "== chấm IN100-C $(date +%T)"
$PY -u eval_general.py --kinds guarded --cache cache_in100c.json --out e3s15_scores_in100c.json F=e3f_in100c_filter.json F15=e3s15_in100c.json A=../p1_9_r6_in100c_bo30_nofp.json SIMP=../p1_8_simp_in100c_full.json
echo "== F15 vs SIMP $(date +%T)"
$PY -u eval_general.py --kinds guarded --cache cache_in100.json --out e3s15_vs_simp_in100.json SIMP=../p1_6a_simp_baseline_in100.json F15=e3s15_in100.json
$PY -u eval_general.py --kinds guarded --cache cache_in100c.json --out e3s15_vs_simp_in100c.json SIMP=../p1_8_simp_in100c_full.json F15=e3s15_in100c.json
echo "== XONG $(date +%T)"
