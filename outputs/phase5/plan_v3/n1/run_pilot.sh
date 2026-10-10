#!/bin/bash
# N1 bước 2 pilot (plan.md mục N1): IN100-B 20 condition đầu, 4 nhánh.
cd /home/tbm/Documents/Input_SIMP_Analyst
B="conda run -n simp python pipeline/phase5_cvae/benchmark_physics_guided_refinement.py --cvae-ckpt outputs/phase5/cvae_v2_finetuned.pt --n-conditions 20 --seed 456 --n-samples 30 --steps 32 --projection-betas 1 4 16 64 --n-boot 2000 --save-images"
O=outputs/phase5/plan_v3/n1
$B --out $O/pilot_B20_A_l0.json > $O/pilot_B20_A_l0.log 2>&1
$B --corner-weight 0.1 --thin-weight 0.1 --out $O/pilot_B20_D_k1.json > $O/pilot_B20_D_k1.log 2>&1
$B --realization-shifts 0 0 0 0.5 0.5 0 0.5 0.5 --n-workers 5 --out $O/pilot_B20_N1_k4.json > $O/pilot_B20_N1_k4.log 2>&1
$B --fe-upsample 2 --out $O/pilot_B20_C_up2.json > $O/pilot_B20_C_up2.log 2>&1
echo ALLDONE
