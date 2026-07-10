"""
Visualization for multi-batch adaptive sampling results.

Produces standalone HTML pages showing:
  - 2D scatter of property space (v12 × v21), coloured by objective
  - Density contour overlay
  - Batch progression (batch 1 points → batch 2 points → ...)
  - Sparse region highlights
  - Per-seed / per-objective breakdown
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np


def _load_batch_results(output_dir: str, batch_id: int) -> List[Dict]:
    """Load results from a batch output directory."""
    json_path = os.path.join(output_dir, f'batch_{batch_id}_results.json')
    if not os.path.exists(json_path):
        return []
    with open(json_path) as f:
        payload = json.load(f)
    return payload.get('results', [])


def _extract_scatter_data(
    results: List[Dict],
    dim_x: str = 'v12',
    dim_y: str = 'v21',
    color_by: str = 'obj_value',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Extract (x, y, color, labels) for plotting.

    Args:
        results: List of result dicts.
        dim_x: Property for x-axis.
        dim_y: Property for y-axis.
        color_by: Property for color mapping.

    Returns:
        x, y, c arrays, and labels list.
    """
    xs, ys, cs, labels = [], [], [], []
    for r in results:
        if not r.get('success'):
            continue
        xv = r.get(dim_x)
        yv = r.get(dim_y)
        cv = r.get(color_by)
        if xv is None or yv is None or cv is None:
            continue
        xs.append(float(xv))
        ys.append(float(yv))
        cs.append(float(cv))
        labels.append(f"sample_{r['sample_id']} seed={r.get('seed','')} obj={cv:.4f}")
    return np.array(xs), np.array(ys), np.array(cs), labels


def _build_html_plot(
    xs: List[float],
    ys: List[float],
    cs: List[float],
    labels: List[str],
    title: str = 'Property Space Coverage',
    dim_x: str = 'v12',
    dim_y: str = 'v21',
    sparse_regions: Optional[List[Dict]] = None,
) -> str:
    """Build standalone HTML with embedded Chart.js scatter.

    Args:
        xs: X data.
        ys: Y data.
        cs: Colour-mapped values (objective).
        labels: Point labels for tooltip.
        title: Chart title.
        dim_x: X axis label.
        dim_y: Y axis label.
        sparse_regions: Optional list of sparse region bounds to highlight.

    Returns:
        Complete HTML string.
    """
    # Color scale: normalized objective (lower = better = highlighted)
    c_array = np.array(cs, dtype=float)
    c_norm = (c_array - c_array.min()) / max(c_array.max() - c_array.min(), 1e-10)
    # Generate RGBA colors: blue (good) -> red (bad)
    colors = []
    for v in c_norm:
        r = int(v * 255)
        b = int((1 - v) * 255)
        colors.append(f'rgba({r}, 0, {b}, 0.8)')

    # Data points JSON
    points_json = json.dumps([
        {'x': float(x), 'y': float(y), 'c': float(c), 'label': lbl, 'color': col}
        for x, y, c, lbl, col in zip(xs, ys, cs, labels, colors)
    ])

    # Sparse regions as rectangles
    sparse_rects = ''
    if sparse_regions:
        rects = []
        for sr in sparse_regions:
            bounds = sr.get('bounds', [])
            if len(bounds) >= 2:
                rects.append({
                    'xMin': bounds[0][0], 'xMax': bounds[0][1],
                    'yMin': bounds[1][0], 'yMax': bounds[1][1],
                })
        sparse_rects = json.dumps(rects)

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1117; color: #e4e6eb; padding: 24px; }}
  .container {{ max-width: 1200px; margin: 0 auto; }}
  h1 {{ font-size: 1.5rem; font-weight: 300; margin-bottom: 8px; color: #f0f2f5; }}
  .subtitle {{ color: #8b8fa3; font-size: 0.9rem; margin-bottom: 24px; }}
  .chart-wrap {{ background: #1a1c23; border-radius: 12px; padding: 20px; position: relative; }}
  canvas {{ width: 100%; height: auto; max-height: 80vh; }}
  .legend {{ display: flex; gap: 16px; flex-wrap: wrap; margin-top: 16px; font-size: 0.8rem; color: #8b8fa3; }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; }}
  .legend-dot {{ width: 10px; height: 10px; border-radius: 50%; }}
  #stats {{ margin-top: 16px; display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }}
  .stat-card {{ background: #1a1c23; border-radius: 8px; padding: 12px 16px; }}
  .stat-label {{ font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px; color: #6b6f82; }}
  .stat-value {{ font-size: 1.3rem; font-weight: 500; margin-top: 4px; }}
</style>
</head>
<body>
<div class="container">
  <h1>{title}</h1>
  <div class="subtitle">{dim_x} vs {dim_y} &middot; Color = objective (lower=better)</div>

  <div class="chart-wrap">
    <canvas id="scatterChart"></canvas>
  </div>

  <div id="stats">
    <div class="stat-card">
      <div class="stat-label">Points</div>
      <div class="stat-value">{len(xs)}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">{dim_x} Range</div>
      <div class="stat-value">{min(xs):.4f} &ndash; {max(xs):.4f}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">{dim_y} Range</div>
      <div class="stat-value">{min(ys):.4f} &ndash; {max(ys):.4f}</div>
    </div>
    {'<div class="stat-card"><div class="stat-label">Sparse Regions</div><div class="stat-value">' + str(len(sparse_regions or [])) + '</div></div>' if sparse_regions else ''}
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels@2"></script>
<script>
const points = {points_json};
const sparseRects = {sparse_rects or '[]'};

// Build annotation for sparse rectangles
const annotations = sparseRects.map((r, i) => ({{
  type: 'box',
  xMin: r.xMin, xMax: r.xMax,
  yMin: r.yMin, yMax: r.yMax,
  backgroundColor: 'rgba(255, 99, 132, 0.08)',
  borderColor: 'rgba(255, 99, 132, 0.3)',
  borderWidth: 1,
  label: {{
    display: true,
    content: 'Sparse ' + (i+1),
    position: 'start',
    color: 'rgba(255, 200, 200, 0.5)',
    font: {{ size: 10 }},
  }},
}}));

new Chart(document.getElementById('scatterChart'), {{
  type: 'scatter',
  data: {{
    datasets: [{{
      label: 'Samples',
      data: points.map(p => ({{ x: p.x, y: p.y }})),
      backgroundColor: points.map(p => p.color),
      pointRadius: 5,
      pointHoverRadius: 8,
      pointHitRadius: 20,
    }}],
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: true,
    aspectRatio: 1.5,
    plugins: {{
      tooltip: {{
        callbacks: {{
          label: function(ctx) {{
            const p = points[ctx.dataIndex];
            return p.label + (p ? ' [' + p.c.toFixed(4) + ']' : '');
          }},
        }},
      }},
      annotation: {{
        annotations: annotations,
      }},
    }},
    scales: {{
      x: {{
        title: {{ display: true, text: '{dim_x}', color: '#8b8fa3' }},
        grid: {{ color: 'rgba(255,255,255,0.05)' }},
        ticks: {{ color: '#8b8fa3' }},
      }},
      y: {{
        title: {{ display: true, text: '{dim_y}', color: '#8b8fa3' }},
        grid: {{ color: 'rgba(255,255,255,0.05)' }},
        ticks: {{ color: '#8b8fa3' }},
      }},
    }},
  }},
  plugins: [ChartDataLabels],
}});
</script>
</body>
</html>'''
    return html


def generate_coverage_html(
    output_dir: str,
    all_results: List[Dict],
    sparse_regions: Optional[List[Dict]] = None,
    title: str = 'Multi-Batch Coverage Analysis',
    dim_x: str = 'v12',
    dim_y: str = 'v21',
) -> str:
    """Generate HTML coverage plot and save to output directory.

    Args:
        output_dir: Where to save the HTML file.
        all_results: Combined batch results.
        sparse_regions: Optional sparse regions to overlay.
        title: Page title.
        dim_x: X property.
        dim_y: Y property.

    Returns:
        Path to generated HTML file.
    """
    xs, ys, cs, labels = _extract_scatter_data(
        all_results, dim_x=dim_x, dim_y=dim_y, color_by='obj_value'
    )

    html = _build_html_plot(
        xs=xs.tolist(),
        ys=ys.tolist(),
        cs=cs.tolist(),
        labels=labels,
        title=title,
        dim_x=dim_x,
        dim_y=dim_y,
        sparse_regions=sparse_regions,
    )

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'coverage_scatter.html')
    with open(path, 'w') as f:
        f.write(html)
    return path


def generate_batch_progression_html(
    output_dir: str,
    batch_dirs: List[str],
    dim_x: str = 'v12',
    dim_y: str = 'v21',
) -> str:
    """Generate HTML with colour-coded batches to show progression.

    Each batch is drawn as a separate dataset with a distinct colour.

    Args:
        output_dir: Where to save.
        batch_dirs: List of batch output directories.
        dim_x: X property.
        dim_y: Y property.

    Returns:
        Path to generated HTML.
    """
    batch_colors = [
        '#4facfe', '#43e97b', '#fa709a', '#a18cd1', '#fbc2eb',
        '#84fab0', '#8fd3f4', '#f6d365', '#fda085', '#e0c3fc',
    ]

    datasets_json = []
    all_xs, all_ys = [], []
    for i, bdir in enumerate(batch_dirs):
        batch_id = i + 1
        results = _load_batch_results(bdir, batch_id)
        xs, ys, _, _ = _extract_scatter_data(results, dim_x, dim_y)
        if len(xs) == 0:
            continue
        all_xs.extend(xs.tolist())
        all_ys.extend(ys.tolist())
        color = batch_colors[i % len(batch_colors)]
        datasets_json.append({
            'label': f'Batch {batch_id}',
            'data': [{'x': float(x), 'y': float(y)} for x, y in zip(xs, ys)],
            'backgroundColor': color,
            'pointRadius': 5,
        })

    datasets_str = json.dumps(datasets_json)

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Batch Progression — Property Space</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1117; color: #e4e6eb; padding: 24px; }}
  .container {{ max-width: 1200px; margin: 0 auto; }}
  h1 {{ font-size: 1.5rem; font-weight: 300; }}
  .subtitle {{ color: #8b8fa3; font-size: 0.9rem; margin-bottom: 24px; }}
  .chart-wrap {{ background: #1a1c23; border-radius: 12px; padding: 20px; }}
  canvas {{ width: 100%; max-height: 80vh; }}
</style>
</head>
<body>
<div class="container">
  <h1>Batch Progression</h1>
  <div class="subtitle">{dim_x} vs {dim_y} &middot; Each batch in a different colour</div>
  <div class="chart-wrap">
    <canvas id="progressionChart"></canvas>
  </div>
</div>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script>
new Chart(document.getElementById('progressionChart'), {{
    type: 'scatter',
    data: {{ datasets: {datasets_str} }},
    options: {{
        responsive: true,
        maintainAspectRatio: true,
        aspectRatio: 1.5,
        plugins: {{
            tooltip: {{
                callbacks: {{
                    label: function(ctx) {{
                        return ctx.dataset.label + ' (' + ctx.parsed.x.toFixed(4) + ', ' + ctx.parsed.y.toFixed(4) + ')';
                    }},
                }},
            }},
        }},
        scales: {{
            x: {{ title: {{ display: true, text: '{dim_x}', color: '#8b8fa3' }}, grid: {{ color: 'rgba(255,255,255,0.05)' }}, ticks: {{ color: '#8b8fa3' }} }},
            y: {{ title: {{ display: true, text: '{dim_y}', color: '#8b8fa3' }}, grid: {{ color: 'rgba(255,255,255,0.05)' }}, ticks: {{ color: '#8b8fa3' }} }},
        }},
    }},
}});
</script>
</body>
</html>'''

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'batch_progression.html')
    with open(path, 'w') as f:
        f.write(html)
    return path