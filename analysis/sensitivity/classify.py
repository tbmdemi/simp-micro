"""
Phân loại tham số dựa trên độ nhạy
=====================================
Dùng SRC và Sobol ST để phân loại tham số thành:
  - Highly sensitive: |SRC| > 0.3 hoặc ST > 0.5
  - Moderately sensitive: |SRC| > 0.1 hoặc ST > 0.2
  - Locally sensitive: Chỉ Sobol S1 > 0.1 (tương tác bậc thấp)
  - Not sensitive: Phần còn lại
"""

from typing import Dict, List, Optional, Tuple


def classify_parameters(
    src_coef: Dict[str, float],
    sobol_st: Dict[str, float],
    sobol_s1: Dict[str, float],
    src_threshold_high: float = 0.3,
    src_threshold_mod: float = 0.1,
    st_threshold_high: float = 0.5,
    st_threshold_mod: float = 0.2,
    s1_threshold_local: float = 0.1,
) -> Dict[str, Dict]:
    """Phân loại tham số dựa trên SRC và Sobol indices.

    Args:
        src_coef: Dict {param_name: SRC coefficient}.
        sobol_st: Dict {param_name: Sobol total-order index}.
        sobol_s1: Dict {param_name: Sobol first-order index}.
        src_threshold_high: Ngưỡng |SRC| cho "highly sensitive".
        src_threshold_mod: Ngưỡng |SRC| cho "moderately sensitive".
        st_threshold_high: Ngưỡng ST cho "highly sensitive".
        st_threshold_mod: Ngưỡng ST cho "moderately sensitive".
        s1_threshold_local: Ngưỡng S1 cho "locally sensitive".

    Returns:
        Dict {param_name: {'class': str, 'src': float, 'st': float, 's1': float}}.
    """
    all_params = set(src_coef.keys()) | set(sobol_st.keys()) | set(sobol_s1.keys())
    result = {}

    for p in sorted(all_params):
        src = src_coef.get(p)
        st = sobol_st.get(p)
        s1 = sobol_s1.get(p)

        # Bỏ qua nếu thiếu dữ liệu
        if src is None and st is None and s1 is None:
            result[p] = {'class': 'no_data', 'src': None, 'st': None, 's1': None}
            continue

        src_abs = abs(src) if src is not None else 0.0
        st_val = st if st is not None else 0.0
        s1_val = s1 if s1 is not None else 0.0

        if src_abs > src_threshold_high or st_val > st_threshold_high:
            classification = 'highly_sensitive'
        elif src_abs > src_threshold_mod or st_val > st_threshold_mod:
            classification = 'moderately_sensitive'
        elif s1_val > s1_threshold_local:
            classification = 'locally_sensitive'
        else:
            classification = 'not_sensitive'

        result[p] = {
            'class': classification,
            'src': src,
            'st': st,
            's1': s1,
        }

    return result


def summarize_classification(classification: Dict[str, Dict]) -> Dict[str, List[str]]:
    """Tổng hợp kết quả phân loại thành các nhóm.

    Args:
        classification: Dict từ classify_parameters().

    Returns:
        Dict {class_name: [param_names]}.
    """
    summary: Dict[str, List[str]] = {}
    for p, info in classification.items():
        cls = info['class']
        if cls not in summary:
            summary[cls] = []
        summary[cls].append(p)
    return summary