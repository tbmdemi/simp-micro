"""
Phân tích độ nhạy (Sensitivity Analysis) trên dữ liệu Phase 1 Screening.

Module này thay thế `sensitivity_analysis.py` cũ (demo dùng LHS + toy model)
bằng cách đọc dữ liệu thật từ Phase 1 (outputs/pipeline/phase1/) và tính:
  - Standardized Regression Coefficients (SRC)
  - Sobol indices (qua surrogate model)
  - ANOVA (one-way + interaction)
  - Phân loại tham số (highly / locally / not sensitive)

Package con:
  - regression.py  : SRC
  - sobol.py       : Sobol indices via Gaussian Process surrogate
  - anova.py       : One-way & two-way ANOVA
  - classify.py    : Parameter classification
  - visualize.py   : Plotting helpers
  - runner.py      : Orchestrator
"""