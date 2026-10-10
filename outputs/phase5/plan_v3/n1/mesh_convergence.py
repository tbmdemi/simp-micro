"""Hội tụ lưới của 2 cách biểu diễn cùng thiết kế 50²: bậc thang (kron) và
bo góc nhẹ (σ=0,25) - để biết khe 50²↔200² là lỗi rời rạc thật hay do cách
biểu diễn biên. Chạy tuần tự (1 tiến trình) để nhẹ RAM."""
import json
import sys

import numpy as np
from eval_pilot import realize

sys.path.insert(0, "/home/tbm/Documents/Input_SIMP_Analyst/pipeline/phase5_cvae")
from verify_fe import evaluate_density_field, resize_to_fe_grid  # noqa: E402

d = json.load(open("../p1_7_c5_in100_bo30_nofp.json"))["per_condition"]
out = {}
for i in [0, 9, 27, 45, 72, 90]:
    img = (np.asarray(d[i]["refined_image"], float) > 0.5).astype(float)
    d50 = (resize_to_fe_grid(img, 50, 50) > 0.5).astype(float)
    row = {"target": d[i]["target_v12"]}
    for n in (50, 100, 200, 400):
        row[f"kron{n}"] = evaluate_density_field(np.kron(d50, np.ones((n // 50, n // 50))))[0]
        row[f"round{n}"] = evaluate_density_field(realize(d50, n, sigma=0.25))[0]
    out[i] = row
    print(i, {k: round(v, 4) for k, v in row.items()}, flush=True)
json.dump(out, open("mesh_convergence.json", "w"), indent=1)
