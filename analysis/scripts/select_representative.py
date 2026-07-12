"""
select_representative.py
──────────────────────────
Quét toàn bộ output Phase 1 LHS Screening, kết hợp metrics từ CSV +
đặc trưng ảnh để xếp hạng seed × sample × iter cho 5 tiêu chí:
  best, worst, complex, simple, stable.

Usage:
    python -m analysis.scripts.select_representative \\
        --phase1_dir outputs/pipeline/phase1 \\
        --objective auxetic --top_n 3

    python -m analysis.scripts.select_representative \\
        --phase1_dir outputs/pipeline/phase1 \\
        --objective all --top_n 3
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from PIL import Image
    PIL_OK = True
except ImportError:
    PIL_OK = False

try:
    from scipy import ndimage
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

# ─── Constants ───────────────────────────────────────────────────────────────
OBJECTIVES = ['auxetic']

SEEDS = [
    'circle', 'square', 'hourglass', 'four_circle', 'hexagonal',
    'nine_circle', 'cross_rectangular', 'grid_circular_voids',
    'small_square_cross', 'circle_half_quarter',
]


# ─────────────────────────────────────────────────────────────────
#  I. CSV log loading
# ─────────────────────────────────────────────────────────────────

def _safe_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float('nan')


def load_csv_log(csv_path: Path) -> Optional[Dict]:
    """Đọc simp_log.csv - trả về dict metrics cuối cùng."""
    if not csv_path.exists():
        return None
    rows = []
    try:
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except Exception:
        return None
    if not rows:
        return None

    last = rows[-1]

    objectives = [_safe_float(r.get('objective', 'nan')) for r in rows]
    obj_vals = [v for v in objectives if not np.isnan(v)]

    tail = obj_vals[-20:] if len(obj_vals) >= 20 else obj_vals
    obj_std = float(np.std(tail)) if tail else float('nan')

    return {
        'final_obj': _safe_float(last.get('objective')),
        'final_v12': _safe_float(last.get('v12')),
        'final_v21': _safe_float(last.get('v21')),
        'final_volume': _safe_float(last.get('volume')),
        'n_iters': len(rows),
        'obj_std': obj_std,
        'obj_history': obj_vals,
    }


# ─────────────────────────────────────────────────────────────────
#  II. Image feature extraction
# ─────────────────────────────────────────────────────────────────

def extract_image_features(img_path: Path) -> Optional[Dict]:
    """Trích xuất đặc trưng hình ảnh từ density map PNG."""
    if not PIL_OK or not img_path.exists():
        return None

    try:
        img = Image.open(img_path).convert('L')
        arr = np.array(img, dtype=np.float32) / 255.0
    except Exception:
        return None

    contrast = float(np.std(arr))
    near_solid = float(np.mean(arr > 0.7))
    near_void = float(np.mean(arr < 0.3))
    clarity = near_solid + near_void

    gx = np.gradient(arr, axis=1)
    gy = np.gradient(arr, axis=0)
    edge_mag = np.sqrt(gx**2 + gy**2)
    edge_density = float(np.mean(edge_mag))

    mid_mask = (arr > 0.3) & (arr < 0.7)
    gray_frac = float(np.mean(mid_mask))

    n_regions = 0
    if SCIPY_OK:
        binary = arr > 0.5
        labeled, n_regions = ndimage.label(binary)
        min_size = int(0.005 * arr.size)
        sizes = ndimage.sum(binary, labeled, range(1, n_regions + 1))
        n_regions = int(np.sum(np.array(sizes) >= min_size))

    return {
        'contrast': contrast,
        'clarity': clarity,
        'edge_density': edge_density,
        'gray_frac': gray_frac,
        'n_regions': n_regions,
        'solid_frac': float(np.mean(arr)),
        'img_path': str(img_path),
    }


# ─────────────────────────────────────────────────────────────────
#  III. Image file discovery
# ─────────────────────────────────────────────────────────────────

def find_images_in_sample(sample_dir: Path) -> List[Tuple[int, Path]]:
    """Tìm tất cả density_iter_*.png, trả về list (iter_num, path)."""
    imgs = []
    pattern = re.compile(r'density_iter_(\d+)\.png', re.IGNORECASE)
    for f in sample_dir.iterdir():
        m = pattern.match(f.name)
        if m:
            imgs.append((int(m.group(1)), f))
    imgs.sort(key=lambda x: x[0])
    return imgs


def get_final_image(sample_dir: Path) -> Optional[Tuple[int, Path]]:
    """Lấy ảnh iteration cuối cùng trong sample."""
    imgs = find_images_in_sample(sample_dir)
    return imgs[-1] if imgs else None


# ─────────────────────────────────────────────────────────────────
#  IV. Phase 1 scanning
# ─────────────────────────────────────────────────────────────────

def scan_phase1(phase1_dir: Path, objective: str,
                seeds: List[str]) -> List[Dict]:
    """Quét phase1_dir/<seed>/<objective>/sample_*/, thu thập metrics."""
    records = []

    for seed in seeds:
        obj_dir = phase1_dir / seed / objective
        if not obj_dir.exists():
            obj_dir = phase1_dir / objective / seed
        if not obj_dir.exists():
            continue

        sample_dirs = sorted([
            d for d in obj_dir.iterdir()
            if d.is_dir() and d.name.startswith('sample_')
        ])

        for sample_dir in sample_dirs:
            m = re.match(r'sample_(\d+)', sample_dir.name)
            sample_id = int(m.group(1)) if m else -1

            csv_path = sample_dir / 'simp_log.csv'
            csv_data = load_csv_log(csv_path)

            final_img = get_final_image(sample_dir)
            iter_num, img_path = final_img if final_img else (None, None)
            img_features = extract_image_features(img_path) if img_path else None

            rec = {
                'seed': seed,
                'objective': objective,
                'sample_id': sample_id,
                'sample_dir': str(sample_dir),
                'final_obj': csv_data['final_obj'] if csv_data else float('nan'),
                'final_v12': csv_data['final_v12'] if csv_data else float('nan'),
                'final_v21': csv_data['final_v21'] if csv_data else float('nan'),
                'final_volume': csv_data['final_volume'] if csv_data else float('nan'),
                'n_iters': csv_data['n_iters'] if csv_data else None,
                'obj_std': csv_data['obj_std'] if csv_data else float('nan'),
                'contrast': img_features['contrast'] if img_features else float('nan'),
                'clarity': img_features['clarity'] if img_features else float('nan'),
                'edge_density': img_features['edge_density'] if img_features else float('nan'),
                'gray_frac': img_features['gray_frac'] if img_features else float('nan'),
                'n_regions': img_features['n_regions'] if img_features else 0,
                'solid_frac': img_features['solid_frac'] if img_features else float('nan'),
                'iter': iter_num,
                'image_path': str(img_path) if img_path else None,
            }
            records.append(rec)

    return records


# ─────────────────────────────────────────────────────────────────
#  V. Ranking criteria
# ─────────────────────────────────────────────────────────────────

def rank_records(records: List[Dict], top_n: int = 3) -> Dict:
    """Xếp hạng theo best, worst, complex, simple, stable."""
    valid = [r for r in records if not np.isnan(r['final_obj'])]
    if not valid:
        return {}

    def top(lst, key_fn, n, reverse=True):
        return sorted(lst, key=key_fn, reverse=reverse)[:n]

    # Với auxetic: final_obj càng cao (≥0) càng tốt
    best = top(valid, lambda r: r['final_obj'], top_n, reverse=True)
    worst = top(valid, lambda r: r['final_obj'], top_n, reverse=False)

    max_edge = max(r['edge_density'] for r in valid) or 1.0
    complex_score = lambda r: (
        r['n_regions'] * 0.6 + (r['edge_density'] / max_edge) * 0.4
    )
    complex_ = top(valid, complex_score, top_n, reverse=True)

    simple_score = lambda r: (
        (1.0 / (r['n_regions'] + 1)) * 0.5
        + r['clarity'] * 0.3
        + (1.0 - r['edge_density'] / max_edge) * 0.2
    )
    simple_ = top(valid, simple_score, top_n, reverse=True)

    max_iters = max((r['n_iters'] or 0) for r in valid) or 1
    max_std = max(r['obj_std'] for r in valid
                   if not np.isnan(r['obj_std'])) or 1.0
    stable_score = lambda r: -(
        (r['obj_std'] / max_std if not np.isnan(r['obj_std']) else 1.0) * 0.6
        + ((r['n_iters'] or max_iters) / max_iters) * 0.4
    )
    stable_ = top(valid, stable_score, top_n, reverse=True)

    return {
        'best': best,
        'worst': worst,
        'complex': complex_,
        'simple': simple_,
        'stable': stable_,
    }


# ─────────────────────────────────────────────────────────────────
#  VI. Output formatting
# ─────────────────────────────────────────────────────────────────

def format_record(r: Dict) -> Dict:
    """Rút gọn record để xuất JSON / in ra màn hình."""
    def _round(v, d=4):
        return round(v, d) if not np.isnan(v) else None

    return {
        'seed': r['seed'],
        'sample_id': r['sample_id'],
        'iter': r['iter'],
        'image_path': r['image_path'],
        'final_obj': _round(r['final_obj']),
        'final_v12': _round(r['final_v12']),
        'final_volume': _round(r['final_volume']),
        'n_iters': r['n_iters'],
        'obj_std': _round(r['obj_std']),
        'n_regions': r['n_regions'],
        'edge_density': _round(r['edge_density']),
        'clarity': _round(r['clarity']),
    }


def print_rankings(rankings: Dict, objective: str) -> None:
    """In kết quả xếp hạng ra terminal."""
    labels = {
        'best': '🏆 BEST   - objective tốt nhất',
        'worst': '🔻 WORST  - objective tệ nhất (converged)',
        'complex': '🔬 COMPLEX - topology phức tạp nhất',
        'simple': '⬜ SIMPLE  - topology đơn giản nhất',
        'stable': '📈 STABLE  - hội tụ ổn định nhất',
    }
    print(f'\n{"═"*64}')
    print(f'  Phase 1 Image Rankings  |  objective: {objective.upper()}')
    print(f'{"═"*64}')

    for key, label in labels.items():
        items = rankings.get(key, [])
        print(f'\n{label}')
        header = f'  {"SEED":<22} {"SAMPLE":>7} {"ITER":>6}  {"OBJ":>10}  {"N_REG":>6}  {"STD":>8}'
        rule = f'  {"─"*22} {"─"*7} {"─"*6}  {"─"*10}  {"─"*6}  {"─"*8}'
        print(header)
        print(rule)
        for r in items:
            obj_s = f'{r["final_obj"]:+.4f}' if r['final_obj'] is not None else '  N/A'
            std_s = f'{r["obj_std"]:.4f}' if r['obj_std'] is not None else '  N/A'
            nreg_s = str(r['n_regions']) if r['n_regions'] is not None else '  N/A'
            print(f'  {r["seed"]:<22} {r["sample_id"]:>7} {r["iter"]:>6}  {obj_s:>10}  {nreg_s:>6}  {std_s:>8}')
            print(f'    → {r["image_path"]}')


# ─────────────────────────────────────────────────────────────────
#  VII. CLI
# ─────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Tìm ảnh đại diện từ Phase 1 LHS output'
    )
    p.add_argument('--phase1_dir', type=str, default='outputs/pipeline/phase1')
    p.add_argument('--objective', type=str, default='auxetic',
                   choices=OBJECTIVES + ['all'])
    p.add_argument('--seeds', type=str, default='all')
    p.add_argument('--top_n', type=int, default=3)
    p.add_argument('--out_json', type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    phase1_dir = Path(args.phase1_dir)

    if not phase1_dir.exists():
        print(f'[ERROR] Không tìm thấy thư mục: {phase1_dir}')
        sys.exit(1)

    seeds = SEEDS if args.seeds == 'all' else [s.strip()
                                                for s in args.seeds.split(',')]
    objectives = OBJECTIVES if args.objective == 'all' else [args.objective]

    all_output = {}

    for obj in objectives:
        records = scan_phase1(phase1_dir, obj, seeds)
        if not records:
            print(f'[WARN] Không tìm thấy dữ liệu cho objective={obj}')
            continue

        rankings = rank_records(records, top_n=args.top_n)
        formatted = {
            k: [format_record(r) for r in v] for k, v in rankings.items()
        }
        all_output[obj] = formatted
        print_rankings(formatted, obj)

    out_path = args.out_json or f'{"_".join(objectives)}_representative.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_output, f, indent=2, ensure_ascii=False)
    print(f'\n[OK] Kết quả lưu tại: {out_path}')


if __name__ == '__main__':
    main()
