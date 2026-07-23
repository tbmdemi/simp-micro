"""
Module tạo báo cáo HTML cho kết quả phân tích SIMP.

Tạo một báo cáo HTML tự chứa với các biểu đồ hội tụ,
chỉ số chất lượng hình ảnh và bảng phân loại.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_classification_table_html(df: pd.DataFrame) -> str:
    """
    Tạo bảng HTML từ DataFrame phân loại.

    Args:
        df (pd.DataFrame): DataFrame với các cột Shape, Poisson_v12, Poisson_v21,
            Classification, Objective, Volume, Iterations.

    Returns:
        str: Chuỗi HTML của bảng.
    """
    if df.empty:
        return '<p>Không có dữ liệu.</p>'

    rows_html = ''
    for _, row in df.iterrows():
        cls = row.get('Classification', '')
        cls_class = 'auxetic' if cls == 'Auxetic' else 'conventional'
        rows_html += f'''<tr>
            <td>{row.get('Shape', '')}</td>
            <td>{row.get('Poisson_v12', 'N/A'):.4f}</td>
            <td>{row.get('Poisson_v21', 'N/A'):.4f}</td>
            <td class="{cls_class}">{cls}</td>
            <td>{row.get('Objective', 'N/A'):.4f}</td>
            <td>{row.get('Volume', 'N/A'):.3f}</td>
            <td>{int(row.get('Iterations', 0))}</td>
        </tr>\n'''

    return f'''<table class="data-table">
        <thead>
            <tr>
                <th>Hình dạng</th>
                <th>ν₁₂</th>
                <th>ν₂₁</th>
                <th>Phân loại</th>
                <th>Mục tiêu</th>
                <th>Thể tích</th>
                <th>Vòng lặp</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>'''


def generate_image_metrics_table_html(df: pd.DataFrame) -> str:
    """
    Tạo bảng HTML từ DataFrame chỉ số hình ảnh.

    Args:
        df (pd.DataFrame): DataFrame với các cột filename, binary_rate, edge_density,
            noise_ratio, symmetry_lr.

    Returns:
        str: Chuỗi HTML của bảng.
    """
    if df.empty:
        return '<p>Không có dữ liệu hình ảnh.</p>'

    rows_html = ''
    for _, row in df.iterrows():
        rows_html += f'''<tr>
            <td>{row.get('filename', '')}</td>
            <td>{row.get('binary_rate', 0):.4f}</td>
            <td>{row.get('edge_density', 0):.4f}</td>
            <td>{row.get('noise_ratio', 0):.4f}</td>
            <td>{row.get('symmetry_lr', 0):.4f}</td>
        </tr>\n'''

    return f'''<table class="data-table">
        <thead>
            <tr>
                <th>Tên file</th>
                <th>Tỉ lệ nhị phân</th>
                <th>Mật độ cạnh</th>
                <th>Tỉ lệ nhiễu</th>
                <th>Đối xứng (T/P)</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>'''


def generate_html_report(
    classification_df: pd.DataFrame,
    image_metrics_df: pd.DataFrame,
    output_path: str,
    title: str = 'Báo cáo phân tích SIMP',
) -> str:
    """
    Tạo một báo cáo HTML tự chứa hoàn chỉnh.

    Args:
        classification_df (pd.DataFrame): DataFrame bảng phân loại.
        image_metrics_df (pd.DataFrame): DataFrame chỉ số hình ảnh.
        output_path (str): Đường dẫn để lưu file HTML.
        title (str): Tiêu đề báo cáo.

    Returns:
        str: Đường dẫn đến file HTML đã tạo.
    """
    # Tính toán thống kê tóm tắt
    n_shapes = len(classification_df) if not classification_df.empty else 0
    n_auxetic = len(
        classification_df[classification_df['Classification'] == 'Auxetic']
    ) if not classification_df.empty else 0
    n_conventional = n_shapes - n_auxetic

    # Bảng phân loại HTML
    class_table_html = generate_classification_table_html(classification_df)

    # Bảng chỉ số hình ảnh HTML
    img_table_html = generate_image_metrics_table_html(image_metrics_df)

    html = f'''<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
:root {{
    --color-bg: #f5f4f0;
    --color-surface: #ffffff;
    --color-border: #e0ddd5;
    --color-text-primary: #1a1a1a;
    --color-text-secondary: #6b6b6b;
    --color-accent: #2563eb;
    --color-accent-dim: #1d4ed8;
    --color-success: #16a34a;
    --color-warning: #d97706;
}}
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: var(--color-bg);
    color: var(--color-text-primary);
    line-height: 1.6;
    padding: 2rem;
}}
.container {{ max-width: 1200px; margin: 0 auto; }}
header {{
    margin-bottom: 2.5rem;
    padding-bottom: 1.5rem;
    border-bottom: 2px solid var(--color-border);
}}
h1 {{ font-size: 2.5rem; font-weight: 700; letter-spacing: -0.02em; margin-bottom: 0.5rem; }}
h2 {{ font-size: 1.5rem; font-weight: 600; margin: 2rem 0 1rem; color: var(--color-text-primary); }}
.summary-cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 2rem; }}
.card {{
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}}
.card-value {{ font-size: 2.5rem; font-weight: 700; color: var(--color-accent); }}
.card-label {{ font-size: 0.875rem; color: var(--color-text-secondary); margin-top: 0.25rem; }}
.data-table {{
    width: 100%;
    border-collapse: collapse;
    background: var(--color-surface);
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}}
.data-table th {{
    background: var(--color-accent);
    color: white;
    padding: 0.75rem 1rem;
    text-align: left;
    font-weight: 600;
    font-size: 0.875rem;
}}
.data-table td {{
    padding: 0.75rem 1rem;
    border-bottom: 1px solid var(--color-border);
    font-size: 0.9rem;
}}
.data-table tr:last-child td {{ border-bottom: none; }}
.data-table tr:hover td {{ background: #f8f7f4; }}
.auxetic {{ color: var(--color-success); font-weight: 600; }}
.conventional {{ color: var(--color-warning); font-weight: 600; }}
footer {{
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--color-border);
    color: var(--color-text-secondary);
    font-size: 0.875rem;
    text-align: center;
}}
@media (prefers-reduced-motion: reduce) {{
    *, *::before, *::after {{ animation-duration: 0.01ms !important; transition-duration: 0.01ms !important; }}
}}
</style>
</head>
<body>
<div class="container">
<header>
    <h1>{title}</h1>
    <p style="color: var(--color-text-secondary);">
        Được tạo từ kết quả tối ưu hóa hình dạng SIMP
    </p>
</header>

<section>
    <h2>Tóm tắt</h2>
    <div class="summary-cards">
        <div class="card">
            <div class="card-value">{n_shapes}</div>
            <div class="card-label">Tổng số hình dạng</div>
        </div>
        <div class="card">
            <div class="card-value" style="color: var(--color-success);">{n_auxetic}</div>
            <div class="card-label">Auxetic</div>
        </div>
        <div class="card">
            <div class="card-value" style="color: var(--color-warning);">{n_conventional}</div>
            <div class="card-label">Thông thường</div>
        </div>
    </div>
</section>

<section>
    <h2>Bảng phân loại</h2>
    {class_table_html}
</section>

<section>
    <h2>Chỉ số chất lượng hình ảnh</h2>
    {img_table_html}
</section>

<footer>
    <p>Được tạo bởi AuxForge &mdash; v1.1.0</p>
</footer>
</div>
</body>
</html>'''

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    output_path_obj.write_text(html, encoding='utf-8')

    logger.info('Báo cáo đã lưu tại: %s', output_path)
    return str(output_path_obj)