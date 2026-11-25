"""
ACP（自适应通道分片）分组可视化。

单图：任意数据集的通道相关性矩阵 |Pearson|，按 ACP 聚类分组标签重排通道，
使强相关通道聚成沿对角线的块状结构，并用组边界线标出每个 patch。

Usage:
    python acp_visualization.py --data_path <csv1> <csv2> --dataset ETTh2 Electricity \
        --out essay/paper_www/acp_groups.pdf --threshold 0.5 --dpi 600

For every input dataset, this script exports an individual 600 dpi PNG first,
combines the PNGs horizontally, and finally converts that combined PNG to PDF.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.cluster import AgglomerativeClustering


def load_data(data_path):
    """加载任意数据集 CSV (T, C)，去掉可能的 date 列。"""
    df = pd.read_csv(data_path)
    cols = [c for c in df.columns if c != 'date']
    return df[cols].values.astype(float)


def compute_groups(X, threshold):
    """与 tools.channel_groups_from_dataset 相同的 ACP 分组逻辑。"""
    C = X.shape[1]
    corr = np.corrcoef(X.T)            # (C, C)
    sim = np.abs(np.nan_to_num(corr, nan=0.0))
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric='precomputed',
        linkage='average',
        distance_threshold=threshold,
        compute_distances=True,
    )
    labels = clustering.fit_predict(dist)
    if int(labels.max()) + 1 < 2:
        labels = np.arange(C)
    return corr, labels


def _safe_name(name):
    """Return a filename-safe dataset label while retaining readable names."""
    return ''.join(char if char.isalnum() or char in '-_' else '_' for char in name)


def _print_group_summary(labels, dataset_name, threshold):
    M = int(labels.max()) + 1
    print(f'\n=== ACP 分组结果 ({dataset_name}, thr={threshold}) ===')
    print(f'通道数 C={len(labels)}, 组数 M={M}')
    for group_id in range(M):
        members = np.where(labels == group_id)[0]
        print(f'  组 {group_id}: {len(members)} 个通道 {members.tolist()}')


def export_panel(data_path, dataset_name, output_path, threshold=0.5, dpi=600):
    """Export one ACP correlation panel as a high-resolution PNG."""
    X = load_data(data_path)
    corr, labels = compute_groups(X, threshold)
    M = int(labels.max()) + 1

    order = np.argsort(labels, kind='stable')
    reordered = corr[np.ix_(order, order)]
    sorted_labels = labels[order]
    boundaries = []
    for group_id in range(M):
        indices = np.where(sorted_labels == group_id)[0]
        if len(indices) > 0:
            boundaries.append(indices[-1] + 0.5)

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(reordered, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    for boundary in boundaries[:-1]:
        ax.axhline(boundary, color='black', linewidth=1.0)
        ax.axvline(boundary, color='black', linewidth=1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f'{dataset_name} channel correlation (ACP, $M$={M} groups, thr={threshold})'
    )
    colorbar = fig.colorbar(image, ax=ax, shrink=0.8)
    colorbar.set_label('Pearson correlation')
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format='png', dpi=dpi)
    plt.close(fig)
    _print_group_summary(labels, dataset_name, threshold)
    print(f'已保存单图: {output_path}')
    return output_path


def combine_pngs_horizontally(panel_paths, output_path, dpi=600):
    """Combine equally scaled ACP PNG panels horizontally without resampling."""
    if not panel_paths:
        raise ValueError('至少需要一幅 ACP 单图才能拼接。')

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.open(path).convert('RGB') for path in panel_paths]
    try:
        width = sum(image.width for image in images)
        height = max(image.height for image in images)
        combined = Image.new('RGB', (width, height), color='white')
        x_offset = 0
        for image in images:
            combined.paste(image, (x_offset, 0))
            x_offset += image.width
        combined.save(output_path, format='PNG', dpi=(dpi, dpi))
    finally:
        for image in images:
            image.close()
    return output_path


def convert_png_to_pdf(png_path, pdf_path, dpi=600):
    """Embed the assembled PNG in a one-page PDF at its declared resolution."""
    pdf_path = Path(pdf_path)
    with Image.open(png_path) as image:
        image.convert('RGB').save(pdf_path, format='PDF', resolution=dpi)
    return pdf_path


def export_and_combine(data_paths, dataset_names, output_path, threshold=0.5, dpi=600):
    """Export individual PNGs, assemble them, and convert the assembled PNG to PDF."""
    if len(data_paths) != len(dataset_names):
        raise ValueError('数据集路径数量必须与数据集名称数量一致。')

    output_path = Path(output_path)
    if output_path.suffix.lower() != '.pdf':
        raise ValueError('--out 必须使用 .pdf 后缀，以保证拼接过程无损。')

    panel_paths = []
    for data_path, dataset_name in zip(data_paths, dataset_names):
        panel_path = output_path.with_name(
            f'{output_path.stem}_{_safe_name(dataset_name)}.png'
        )
        panel_paths.append(
            export_panel(data_path, dataset_name, panel_path, threshold=threshold, dpi=dpi)
        )

    combined_png = output_path.with_suffix('.png')
    combine_pngs_horizontally(panel_paths, combined_png, dpi=dpi)
    convert_png_to_pdf(combined_png, output_path, dpi=dpi)
    print(f'\n已拼接 {len(panel_paths)} 幅单图: {combined_png}')
    print(f'已由合成 PNG 转换为 PDF: {output_path}')
    return panel_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, nargs='+', required=True,
                        help='一个或多个数据集 CSV 路径，例如 ./datasets/weather/weather.csv')
    parser.add_argument('--out', type=str, default='acp_groups.pdf')
    parser.add_argument('--dataset', type=str, nargs='+', default=None,
                        help='与数据集路径一一对应的图标题名称（默认取文件名）')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='ACP 距离阈值 (与 --merge_threshold 一致)，越大分组越少')
    parser.add_argument('--dpi', type=int, default=600,
                        help='单图 PNG 及最终 PDF 的分辨率，默认 600 dpi')
    parser.add_argument('--panels-only', action='store_true',
                        help='只生成每个数据集的单图 PNG，不在当前环境拼接')
    args = parser.parse_args()

    dataset_names = args.dataset
    if dataset_names is None:
        dataset_names = [Path(data_path).stem for data_path in args.data_path]

    if args.panels_only:
        output_path = Path(args.out)
        if output_path.suffix.lower() != '.pdf':
            parser.error('--out 必须使用 .pdf 后缀。')
        for data_path, dataset_name in zip(args.data_path, dataset_names):
            export_panel(
                data_path,
                dataset_name,
                output_path.with_name(f'{output_path.stem}_{_safe_name(dataset_name)}.png'),
                threshold=args.threshold,
                dpi=args.dpi,
            )
    else:
        export_and_combine(
            data_paths=args.data_path,
            dataset_names=dataset_names,
            output_path=args.out,
            threshold=args.threshold,
            dpi=args.dpi,
        )


if __name__ == '__main__':
    main()
