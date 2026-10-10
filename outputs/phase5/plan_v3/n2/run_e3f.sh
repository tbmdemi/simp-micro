#!/bin/bash
# N2-E3-F (plan.md): xác nhận thiết kế qua bộ lọc σ=1 trên IN100 (chính) + IN100-C (tái lập).
cd /home/tbm/Documents/Input_SIMP_Analyst
PY=/home/tbm/miniconda3/envs/simp/bin/python
B="$PY -u pipeline/phase5_cvae/benchmark_physics_guided_refinement.py --cvae-ckpt outputs/phase5/cvae_v2_finetuned.pt --n-conditions 100 --n-samples 30 --steps 32 --projection-betas 1 4 16 64 --n-boot 10000 --save-images --design-filter-sigma 1.0"
O=outputs/phase5/plan_v3/n2
( /usr/bin/time -f "WALL %e s" $B --seed 123 --out $O/e3f_in100_filter.json > $O/e3f_in100_filter.log 2>&1; echo "IN100 EXIT $?" >> $O/e3f_in100_filter.log ) &
( /usr/bin/time -f "WALL %e s" $B --seed 789 --out $O/e3f_in100c_filter.json > $O/e3f_in100c_filter.log 2>&1; echo "IN100C EXIT $?" >> $O/e3f_in100c_filter.log ) &
wait
echo ALLDONE
