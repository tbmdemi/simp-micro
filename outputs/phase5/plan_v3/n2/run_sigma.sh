#!/bin/bash
# N2-S: ablation σ của bộ lọc thiết kế (plan.md mục N2-S).
cd /home/tbm/Documents/Input_SIMP_Analyst
PY=/home/tbm/miniconda3/envs/simp/bin/python
B="$PY -u pipeline/phase5_cvae/benchmark_physics_guided_refinement.py --cvae-ckpt outputs/phase5/cvae_v2_finetuned.pt --n-conditions 20 --seed 456 --n-samples 30 --steps 32 --projection-betas 1 4 16 64 --n-boot 2000 --save-images"
O=outputs/phase5/plan_v3/n2
for S in 0.5 0.75 1.5; do
  ( $B --design-filter-sigma $S --out $O/pilot_B20_F_s$S.json > $O/pilot_B20_F_s$S.log 2>&1; echo "s$S EXIT $?" >> $O/pilot_B20_F_s$S.log ) &
done
wait
echo "== chấm điểm $(date +%T)"
cd $O
$PY -u eval_general.py --kinds guarded --cache cache_sigma.json --out sigma_scores.json A=../n1/pilot_B20_A_l0.json F_s0.5=pilot_B20_F_s0.5.json F_s0.75=pilot_B20_F_s0.75.json F_s1.0=pilot_B20_F_filter.json F_s1.5=pilot_B20_F_s1.5.json
echo "== XONG $(date +%T)"
