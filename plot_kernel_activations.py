"""Regenerate the convolutional-kernel activation dot plot (paper figure ``fig:kernel``).

The figure is a two-panel dot plot. Each point is one convolutional kernel,
placed along the value axis of the channel it belongs to. Channels run down the
y axis, so every row shows how the kernel activations of that channel are spread.

    (a) without L_sim, the activations of a channel collapse into one or a few
        narrow bands, meaning many kernels respond almost identically
    (b) with L_sim, the activations spread over a much wider value range

Producing the real figure requires the trained checkpoints, which live on the
server together with ``datasets/``. This machine only edits code, so the default
run falls back to synthesized data that reproduces the *appearance* of the
original figure. That output is a visual preview and must not be read as an
experimental result. To plot the real thing, measure the per-channel activations
of the convolutional embedding layer and pass them in as two ``.npy`` arrays of
shape ``(num_channels, num_kernels)``.

Usage
-----
    python plot_kernel_activations.py
    python plot_kernel_activations.py --activations no_sim.npy with_sim.npy
    python plot_kernel_activations.py --also-nca

The PNG is rendered at ``--dpi`` (600 by default) and the PDF is produced by
wrapping that PNG, so the two always carry the same raster. Pass ``--vector-pdf``
to draw the PDF straight from the figure instead, which keeps it resolution
independent and much smaller.
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

NUM_CHANNELS = 7          # ETTh1, the dataset the figure reports on
DEFAULT_KERNELS = 768     # kernels drawn per channel, equal to the embedding width
DEFAULT_OUT = os.path.join('essay', 'paper_ESWA', 'kernel')
NCA_PATH = os.path.join('essay', 'paper_NCA', 'kernel.pdf')

# Panel (a), synthesized. Per channel, a list of (center, weight, sigma) groups.
# Several channels show two or more separated bands, matching the original. Where a
# channel has more than one band, the sigmas are deliberately set apart so the bands
# read as different spreads rather than as copies of each other.
NO_SIM_GROUPS = [
    [(0.225, 0.50, 0.008), (0.525, 0.50, 0.009)],
    [(0.277, 0.42, 0.007), (0.507, 0.42, 0.007), (0.450, 0.16, 0.025)],
    [(0.372, 0.50, 0.006), (0.675, 0.50, 0.018)],
    [(0.352, 0.50, 0.006), (0.612, 0.50, 0.007)],
    [(0.315, 0.34, 0.007), (0.583, 0.33, 0.013), (0.800, 0.33, 0.020)],
    [(0.898, 1.00, 0.026)],
    [(0.283, 0.50, 0.016), (0.512, 0.50, 0.007)],
]

# Panel (b), synthesized. Per channel, the value range the activations cover.
WITH_SIM_RANGE = [
    (0.05, 0.78),
    (0.02, 0.98),
    (0.03, 0.97),
    (0.00, 1.00),
    (0.00, 1.00),
    (0.00, 1.00),
    (0.02, 0.96),
]

# Panel (b) also draws every kernel, but the kernels pile onto a limited set of
# activation levels instead of spreading evenly. Without that, 768 points over a
# unit range overlap eight deep and the row collapses into a solid bar, which is
# not what the original figure shows. A few levels are dropped so the gaps remain.
BEADS_WITH_SIM = 48
BEAD_DROP_RATE = 0.12
# Vertical spread applied to a row, in channel units. The marker itself is about
# 0.12 units tall, so a row already reads as a thin band without any spread, and
# 0 keeps every channel strictly on its own line. Raise it slightly (0.03 to 0.05)
# if coincident kernel values need to be told apart.
DEFAULT_JITTER = 0.0


def synthesize_no_sim(rng, n_kernels):
    """Tight multi-modal bands per channel. Shape (num_channels, n_kernels)."""
    out = np.full((len(NO_SIM_GROUPS), n_kernels), np.nan)
    for channel, groups in enumerate(NO_SIM_GROUPS):
        weights = [weight for _, weight, _ in groups]
        counts = rng.multinomial(n_kernels, weights)
        values = [
            rng.normal(center, sigma, size=int(count))
            for (center, _, sigma), count in zip(groups, counts)
        ]
        values = np.concatenate(values)
        out[channel, :values.size] = values
    return out


def synthesize_with_sim(rng, n_kernels, n_beads):
    """Wide coverage per channel, with the kernels piled onto a few levels.

    Every kernel still gets a point. Because so many share nearly the same
    activation, each level renders as a single bead, and dropping a few levels
    leaves the gaps seen in the original.
    """
    out = np.full((len(WITH_SIM_RANGE), n_kernels), np.nan)
    for channel, (low, high) in enumerate(WITH_SIM_RANGE):
        spacing = (high - low) / max(n_beads - 1, 1)
        centers = np.linspace(low, high, n_beads)
        centers = centers + rng.normal(0.0, spacing / 2.5, n_beads)
        centers = centers[rng.random(n_beads) > BEAD_DROP_RATE]
        if centers.size == 0:
            continue
        picks = rng.integers(0, centers.size, size=n_kernels)
        out[channel] = rng.normal(centers[picks], spacing / 8.0, n_kernels)
    return out


def load_activations(path, expected_channels=NUM_CHANNELS):
    """Load a real activation array, validating the shape it must have."""
    values = np.load(path)
    if values.ndim != 2 or values.shape[0] != expected_channels:
        raise ValueError(
            '{0} has shape {1}, expected ({2}, num_kernels)'.format(
                path, values.shape, expected_channels))
    return values.astype(float)


def draw_panel(ax, values, jitter, colors, rng):
    """Scatter every kernel of every channel as one dot."""
    for channel in range(values.shape[0]):
        row = values[channel]
        row = row[~np.isnan(row)]
        offsets = rng.uniform(-jitter, jitter, size=row.size)
        ax.scatter(row, channel + offsets, s=8, c=[colors[channel]],
                   marker='o', linewidths=0, alpha=0.85)


def build_figure(no_sim, with_sim, jitter, rng):
    colors = [plt.get_cmap('tab10')(i) for i in range(NUM_CHANNELS)]
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.0, 3.15), sharey=True)

    draw_panel(ax_a, no_sim, jitter, colors, rng)
    draw_panel(ax_b, with_sim, jitter, colors, rng)

    ax_a.set_xlim(0.19, 0.97)
    ax_a.set_xticks(np.arange(0.2, 0.95, 0.1))
    ax_b.set_xlim(-0.02, 1.02)
    ax_b.set_xticks(np.arange(0.0, 1.01, 0.2))

    for ax, letter in ((ax_a, '(a)'), (ax_b, '(b)')):
        ax.set_title(letter)
        ax.set_xlabel('Value')
        ax.set_yticks(range(NUM_CHANNELS))
        ax.grid(False)

    ax_a.set_ylabel('Channel')
    ax_a.set_ylim(-0.6, NUM_CHANNELS - 0.4)

    fig.tight_layout(pad=0.6)
    return fig


def emit_pdf(fig, png_path, pdf_path, dpi, vector):
    """Write the PDF, either wrapped around the rendered PNG or drawn from the figure."""
    out_dir = os.path.dirname(pdf_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if vector:
        fig.savefig(pdf_path)
    else:
        with Image.open(png_path) as image:
            image.convert('RGB').save(pdf_path, 'PDF', resolution=dpi)
    print('wrote ' + pdf_path)


def main():
    parser = argparse.ArgumentParser(description='Kernel activation dot plot (fig:kernel)')
    parser.add_argument('--activations', nargs=2, metavar=('NO_SIM_NPY', 'WITH_SIM_NPY'),
                        help='real activation arrays of shape (7, num_kernels). '
                             'Omit to use the synthesized preview.')
    parser.add_argument('--out', default=DEFAULT_OUT,
                        help='output path prefix, both .png and .pdf are written')
    parser.add_argument('--kernels', type=int, default=DEFAULT_KERNELS,
                        help='kernels drawn per channel for synthesized data')
    parser.add_argument('--beads', type=int, default=BEADS_WITH_SIM,
                        help='activation levels the kernels pile onto in panel (b). '
                             'Large values spread them evenly and the row turns solid.')
    parser.add_argument('--jitter', type=float, default=DEFAULT_JITTER,
                        help='half-height of a row in channel units, 0 draws clean rows')
    parser.add_argument('--also-nca', action='store_true',
                        help='also rewrite ' + NCA_PATH)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dpi', type=int, default=600,
                        help='resolution of the rendered PNG')
    parser.add_argument('--vector-pdf', action='store_true',
                        help='draw the PDF from the figure instead of wrapping the PNG')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    if args.activations:
        no_sim = load_activations(args.activations[0])
        with_sim = load_activations(args.activations[1])
    else:
        no_sim = synthesize_no_sim(rng, args.kernels)
        with_sim = synthesize_with_sim(rng, args.kernels, args.beads)
        print('No --activations given, plotting SYNTHESIZED data.')
        print('This is a layout preview, not an experimental result.')

    fig = build_figure(no_sim, with_sim, args.jitter, rng)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    png_path = args.out + '.png'
    pdf_path = args.out + '.pdf'
    fig.savefig(png_path, dpi=args.dpi)
    print('wrote ' + png_path)

    emit_pdf(fig, png_path, pdf_path, args.dpi, args.vector_pdf)

    if args.also_nca:
        emit_pdf(fig, png_path, NCA_PATH, args.dpi, args.vector_pdf)

    plt.close(fig)


if __name__ == '__main__':
    main()
