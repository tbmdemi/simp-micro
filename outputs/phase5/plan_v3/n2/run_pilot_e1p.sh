#!/bin/bash
# N2-E1′ pilot (plan.md mục N2-E1′): F = chỉ thiết kế đã lọc; E1′ = + robust.
cd /home/tbm/Documents/Input_SIMP_Analyst
PY=/home/tbm/miniconda3/envs/simp/bin/python
B="$PY -u pipeline/phase5_cvae/benchmark_physics_guided_refinement.py --cvae-ckpt outputs/phase5/cvae_v2_finetuned.pt --n-conditions 20 --seed 456 --n-samples 30 --steps 32 --projection-betas 1 4 16 64 --n-boot 2000 --save-images --design-filter-sigma 1.0"
O=outputs/phase5/plan_v3/n2
( /usr/bin/time -f "WALL %e s" $B --out $O/pilot_B20_F_filter.json > $O/pilot_B20_F_filter.log 2>&1; echo "F EXIT $?" >> $O/pilot_B20_F_filter.log ) &
( /usr/bin/time -f "WALL %e s" $B --robust-etas 0.25 0.75 --n-workers 2 --out $O/pilot_B20_E1p_robust.json > $O/pilot_B20_E1p_robust.log 2>&1; echo "E1p EXIT $?" >> $O/pilot_B20_E1p_robust.log ) &
wait
echo ALLDONE
