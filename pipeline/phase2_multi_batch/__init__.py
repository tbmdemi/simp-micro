"""
Pipeline đa batch cho adaptive sampling trong SIMP topology optimization.

Quy trình:
  1. Batch 1: Sobol / Optimized LHS (~100-150 điểm) trên không gian đã thu hẹp
  2. Phân tích coverage: scatter plot trong property space (ν₁₂ vs objective)
  3. Phát hiện vùng thưa (sparsity detection)
  4. Batch 2: Directed sampling tập trung vào vùng quan trọng
  5. Batch 3 (nếu cần): Fine-tuning
"""