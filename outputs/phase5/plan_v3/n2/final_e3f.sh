#!/bin/bash
# E3-F chấm điểm: chờ chấm trước A+SIMP (cache) xong, rồi chấm F (guarded) với
# 8 tiến trình và so sánh đầy đủ (plan.md N2-E3-F).
cd /home/tbm/Documents/Input_SIMP_Analyst/outputs/phase5/plan_v3/n2
PY=/home/tbm/miniconda3/envs/simp/bin/python
echo "== chờ chấm trước A+SIMP $(date +%T)"
until grep -q "PRESCORE DONE" prescore.log; do sleep 10; done
echo "== IN100 (chính) $(date +%T)"
$PY -u eval_general.py --kinds guarded --cache cache_in100.json --out e3f_scores_in100.json A=../p1_7_c5_in100_bo30_nofp.json F=e3f_in100_filter.json SIMP=../p1_6a_simp_baseline_in100.json
echo "== IN100-C (tái lập) $(date +%T)"
$PY -u eval_general.py --kinds guarded --cache cache_in100c.json --out e3f_scores_in100c.json A=../p1_9_r6_in100c_bo30_nofp.json F=e3f_in100c_filter.json SIMP=../p1_8_simp_in100c_full.json
echo "== F so với SIMP, IN100 $(date +%T)"
$PY -u eval_general.py --kinds guarded --cache cache_in100.json --out e3f_scores_in100_vs_simp.json SIMP=../p1_6a_simp_baseline_in100.json F=e3f_in100_filter.json
echo "== F so với SIMP, IN100-C $(date +%T)"
$PY -u eval_general.py --kinds guarded --cache cache_in100c.json --out e3f_scores_in100c_vs_simp.json SIMP=../p1_8_simp_in100c_full.json F=e3f_in100c_filter.json
echo "== XONG $(date +%T)"
