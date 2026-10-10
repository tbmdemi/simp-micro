#!/bin/bash
# N2 pilot (plan.md mục N2): E1 robust formulation, E2 đa độ phân giải.
# Chạy song song, log không đệm (python -u) để xem tiến độ trực tiếp.
cd /home/tbm/Documents/Input_SIMP_Analyst
PY=/home/tbm/miniconda3/envs/simp/bin/python
B="$PY -u pipeline/phase5_cvae/benchmark_physics_guided_refinement.py --cvae-ckpt outputs/phase5/cvae_v2_finetuned.pt --n-conditions 20 --seed 456 --n-samples 30 --steps 32 --projection-betas 1 4 16 64 --n-boot 2000 --save-images"
O=outputs/phase5/plan_v3/n2
( /usr/bin/time -f "WALL %e s" $B --robust-etas 0.25 0.75 --robust-sigma 1.0 --n-workers 2 --out $O/pilot_B20_E1_robust.json > $O/pilot_B20_E1_robust.log 2>&1; echo "E1 EXIT $?" >> $O/pilot_B20_E1_robust.log ) &
( /usr/bin/time -f "WALL %e s" $B --fe-upsample 2 --fe-upsample-last-only --out $O/pilot_B20_E2_last100.json > $O/pilot_B20_E2_last100.log 2>&1; echo "E2 EXIT $?" >> $O/pilot_B20_E2_last100.log ) &
wait
echo ALLDONE
