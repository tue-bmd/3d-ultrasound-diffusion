"""Pre/post-processing, subsampling masks, baselines and plotting for 3D volumes.

Throughout, volumes have shape ``(frames, azimuth, depth, elevation, 1)``. Every ``[t, a]`` slice
is a B-plane (depth x elevation). Elevation is the subsampled axis.
"""

import matplotlib.pyplot as plt
import numpy as np
from keras import ops
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from zea import log
from zea.agent.selection import EquispacedLines
from zea.display import scan_convert_2d
from zea.func import interpolate_data, translate, vmap
from zea.visualize import plot_frustum_vertices, set_mpl_style

TITLES = {
    "masks": "Acquisitions",
    "nearest": "Nearest",
    "linear": "Linear",
    "supervised": "U-Net",
    "diffusion": "Diffusion",
    "clean": "Ground truth",
}


def set_plot_style():
    """Use the zea matplotlib style."""
    set_mpl_style()


MASK_CMAP = LinearSegmentedColormap.from_list("mask", ["#3a3a3a", "#1f77b4"])
"""Not acquired (dark gray) -> acquired (blue)."""


def preprocess(data, image_shape, image_range, normalization_range):
    """Resize B-planes to ``image_shape`` and normalize.

    Args:
        data: ``(frames, azimuth, depth, elevation)``.
        image_shape: ``(depth, elevation)`` expected by the models.

    Returns:
        ``(frames, azimuth, *image_shape, 1)`` float32 tensor in ``normalization_range``.
    """
    data = ops.convert_to_tensor(data, "float32")
    frames, azimuth = ops.shape(data)[:2]
    data = ops.reshape(data, (-1, *ops.shape(data)[2:], 1))
    data = ops.image.resize(data, tuple(image_shape))
    data = translate(ops.clip(data, *image_range), image_range, normalization_range)
    return ops.reshape(data, (frames, azimuth, *image_shape, 1))


def get_masks(image_shape, num_frames, subsampling_rate, line_width=2):
    """Equispaced elevation-plane masks that shift every frame.

    The requested rate is rounded to the closest rate that gives equally spaced lines.

    Returns:
        masks: ``(frames, 1, depth, elevation, 1)``, broadcastable over azimuth.
        rate: the effective subsampling rate.
    """
    n_elevation = image_shape[1]
    n_possible = n_elevation // line_width
    requested = int(np.round(n_elevation * subsampling_rate / line_width))
    factors = [i for i in range(1, n_possible + 1) if n_possible % i == 0]
    n_lines = min(factors, key=lambda x: abs(x - requested))
    rate = float(np.round(n_lines / n_possible, 2))
    if rate != subsampling_rate:
        log.warning(f"Subsampling rate {subsampling_rate} not possible, using {rate} instead.")
    log.info(
        f"Acquiring {log.green(n_lines * line_width)} of {n_elevation} elevation planes "
        f"(rate {log.green(rate)}, line width {line_width})."
    )

    sampler = EquispacedLines(
        n_actions=n_lines,
        n_possible_actions=n_possible,
        img_width=image_shape[1],
        img_height=image_shape[0],
    )
    masks, lines = [], None
    for _ in range(num_frames):
        if lines is None:
            lines, mask = sampler.initial_sample_stateless()
        else:
            lines, mask = sampler.sample_stateless(lines)
        masks.append(mask)
    masks = ops.concatenate(masks)  # (frames, depth, elevation)
    return ops.cast(masks[:, None, ..., None], "float32"), rate


def interpolate(measurements, masks, order=1):
    """Linear (``order=1``) or nearest (``order=0``) interpolation along elevation."""
    masks = ops.broadcast_to(masks, ops.shape(measurements))
    out = []
    for measurement, mask in zip(measurements[..., 0], masks[..., 0]):  # loop over frames
        mask = ops.cast(mask, "bool")
        out.append(
            ops.stack(
                [
                    interpolate_data(m[k], k, order=order, axis=-1)
                    for m, k in zip(measurement, mask)  # loop over azimuth
                ]
            )
        )
    return ops.stack(out)[..., None]


def predict_supervised(model, measurements, masks, batch_size=None):
    """U-Net baseline: missing planes are set to the center of the input range."""
    center = sum(model.input_range) / 2
    inputs = ops.where(ops.cast(masks, "bool"), measurements, center)
    shape = ops.shape(inputs)
    inputs = ops.reshape(inputs, (-1, *shape[2:]))
    predict = vmap(
        lambda x: model.predict(x, verbose=False),
        batch_size=batch_size,
        fn_supports_batch=True,
        disable_jit=True,
    )
    return ops.reshape(predict(inputs), shape)


def b_planes(volumes):
    """``(..., azimuth, depth, elevation, 1)`` -> ``(N, depth, elevation, 1)``."""
    return ops.reshape(volumes, (-1, *ops.shape(volumes)[-3:]))


def a_planes(volumes):
    """``(frames, azimuth, depth, elevation, 1)`` -> ``(frames, elevation, depth, azimuth, 1)``."""
    return ops.transpose(volumes, (0, 3, 2, 1, 4))


def to_display(images, image_range, grid=None, plane="B"):
    """2D planes ``(N, depth, width)`` -> scan converted ``[0, 255]`` images (NaN outside).

    Args:
        plane: "B" (depth x elevation, uses ``phi``) or "A" (depth x azimuth, uses ``theta``).
    """
    images = ops.convert_to_numpy(images).astype("float32")
    images = translate(np.clip(images, *image_range), image_range, (0, 255))
    if grid is None:
        return np.asarray(images)
    theta_range = grid["phi_range"] if plane == "B" else grid["theta_range"]
    images, _ = scan_convert_2d(  # expects (..., rho, theta) = (..., depth, angle)
        images, rho_range=grid["rho_range"], theta_range=theta_range, fill_value=np.nan
    )
    return np.asarray(images)


ACQUIRED_STYLE = {"color": "#f4a6c0", "linestyle": "--", "linewidth": 0.8}
"""Acquired elevation planes in the frustum plot (as in the paper's overview figure)."""
SLICE_STYLE = {"color": "#2ca02c", "linestyle": "-", "linewidth": 1.8}
"""Displayed B-plane and A-plane (cross sections) in the frustum plot."""


def plot_frustum(ax, grid, acquired_phi, b_plane_theta, a_plane_phi):
    """3D frustum with the acquired elevation planes and the displayed cross sections.

    Args:
        ax: 3D matplotlib axis.
        grid: frustum ``{rho,theta,phi}_range``.
        acquired_phi: elevation angles of the acquired planes.
        b_plane_theta: azimuth angle of the displayed B-plane.
        a_plane_phi: elevation angle of the displayed A-plane.
    """
    ranges = (grid["rho_range"], grid["theta_range"], grid["phi_range"])
    outline = {"color": plt.rcParams["text.color"], "linewidth": 0.8, "alpha": 0.7}
    common = dict(fig=ax.figure, ax=ax, frustum_style=outline)
    plot_frustum_vertices(*ranges, phi_plane=list(acquired_phi), phi_style=ACQUIRED_STYLE, **common)
    plot_frustum_vertices(
        *ranges,
        phi_plane=[a_plane_phi],
        theta_plane=[b_plane_theta],
        phi_style=SLICE_STYLE,
        theta_style=SLICE_STYLE,
        **common,
    )
    ax.view_init(elev=20, azim=-55)
    handles = [
        Line2D([], [], **ACQUIRED_STYLE, label="acquired"),
        Line2D([], [], **SLICE_STYLE, label="shown"),
    ]
    ax.legend(handles=handles, loc="lower center", fontsize=8, frameon=False, ncol=2)


def plot_comparison(
    rows, row_labels, save_path=None, dpi=200, aspect="auto", frustum=None, suptitle=None
):
    """Grid of images: one row per plane type, one column per method.

    Args:
        rows: list of ``{title: 2D image}`` dicts (same keys for every row).
        row_labels: label for every row.
        frustum: optional kwargs for :func:`plot_frustum`, drawn in an extra first column.
        suptitle: optional title, placed above the frustum if there is one.
    """
    titles = list(rows[0])
    background = plt.rcParams["figure.facecolor"]  # area outside the scan sector
    n_rows, n_cols = len(rows), len(titles) + (frustum is not None)
    widths = [2.2] * (frustum is not None) + [1] * len(titles)  # larger frustum column
    fig = plt.figure(figsize=(1.6 * sum(widths), 1.8 * n_rows))
    gs = fig.add_gridspec(n_rows, n_cols, width_ratios=widths)
    offset = 0
    if frustum is not None:
        ax3d = fig.add_subplot(gs[:, 0], projection="3d")
        plot_frustum(ax3d, **frustum)
        if suptitle is not None:  # above the frustum instead of above the whole figure
            ax3d.set_title(suptitle, y=0.95)
            suptitle = None
        offset = 1
    for r, (row, label) in enumerate(zip(rows, row_labels)):
        for c, title in enumerate(titles):
            ax = fig.add_subplot(gs[r, c + offset])
            if title == TITLES["masks"]:
                cmap = MASK_CMAP.copy()
                cmap.set_bad(background)
                ax.imshow(row[title], cmap=cmap, vmin=0, vmax=1, aspect=aspect)
            else:
                cmap = plt.get_cmap("gray").copy()
                cmap.set_bad(background)
                ax.imshow(row[title], cmap=cmap, vmin=0, vmax=255, aspect=aspect)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0:
                ax.set_title(title)
            if c == 0:
                ax.set_ylabel(label)
    if suptitle is not None:
        fig.suptitle(suptitle)
    fig.tight_layout(pad=0.3)
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        log.success(f"Saved {log.yellow(save_path)}")
    return fig
