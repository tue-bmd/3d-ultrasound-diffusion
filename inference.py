"""Reconstruct a subsampled 3D ultrasound sequence and compare against baselines.

Elevation planes are subsampled with equispaced (per-frame shifting) masks. The missing planes are
reconstructed with SeqDiff + DPS guidance with a TV prior across B-planes ("diffusion") and
compared to nearest / linear interpolation (and optionally a supervised U-Net). Image quality
metrics are printed and saved, together with figures.

Example (after training a model with train.py):
    python inference.py --data /path/to/data/test
"""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("MPLBACKEND", "Agg")  # figures are only saved, never shown

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import imageio.v2 as imageio
import jax
import matplotlib.pyplot as plt
import numpy as np
import tyro
import zea
from keras import ops
from zea import log
from zea.func import translate
from zea.internal.operators import InpaintingOperator
from zea.metrics import Metrics
from zea.models.diffusion import DiffusionModel
from zea.models.unet import UNet

import guidance  # noqa: F401  (registers the "dps_smooth_B_plane" guidance)
import paths
from data import load_volume, resolve_file
from seqdiff import SeqDiff
from utils import (
    TITLES,
    a_planes,
    b_planes,
    get_masks,
    interpolate,
    plot_comparison,
    predict_supervised,
    preprocess,
    set_plot_style,
    to_display,
)

Method = Literal["diffusion", "supervised", "linear", "nearest"]

# SSIM settings used in the paper
SSIM_KWARGS = dict(filter_size=22, filter_sigma=3.0, k1=0.05, k2=0.09)


@dataclass
class Args:
    data: str = str(paths.TEST_DIR)
    """Local folder / hdf5 file, or ``hf://<repo>/<subdir>``."""
    file: Optional[str] = None
    """File (relative to --data) to reconstruct. A random file is picked if not set."""
    methods: List[Method] = field(default_factory=lambda: ["diffusion", "linear", "nearest"])
    """Methods to compare. "supervised" also requires --supervised-model."""
    diffusion_model: str = paths.DIFFUSION_MODEL
    """Local preset folder (default: the output of train.py) or ``hf://<repo>``."""
    supervised_model: Optional[str] = paths.SUPERVISED_MODEL
    """Optional supervised U-Net baseline (zea preset folder or ``hf://<repo>``)."""
    output_dir: Path = paths.RESULTS_DIR / "inference"

    subsampling_rate: float = 0.33
    """Fraction of elevation planes that is acquired."""
    line_width: int = 2
    """Number of adjacent elevation planes acquired together."""
    num_frames: int = 5
    buffer_length: int = 1
    """Number of initial fully-sampled frames used to initialize SeqDiff (excluded from metrics)."""
    db_range: Tuple[float, float] = (-60.0, 0.0)
    """Dynamic range for zea-format (dB) files."""

    num_steps: int = 200
    """Number of steps of the full diffusion trajectory."""
    min_num_steps: int = 50
    """SeqDiff: steps actually run per frame, starting from the previous frame's estimate."""
    omega: float = 35.0
    """Data consistency weight."""
    alpha: float = 0.005
    """TV smoothness weight across B-planes (dps_smooth_B_plane only)."""
    init_omega: Optional[float] = None
    """Data consistency weight for a first frame sampled from scratch (only used when
    buffer_length is 0). Defaults to omega."""
    guidance: Literal["dps_smooth_B_plane", "dps"] = "dps_smooth_B_plane"
    num_particles: int = 1
    batch_size: Optional[int] = None
    """Number of B-planes sampled in parallel (None: all). Lower to reduce memory."""

    seed: int = 12
    device: str = "auto:1"
    lpips: bool = True
    """Also compute LPIPS (downloads the LPIPS weights)."""
    verbose: bool = False


def load_models(args: Args):
    diffusion = supervised = None
    if "diffusion" in args.methods:
        if not args.diffusion_model.startswith("hf://") and not Path(args.diffusion_model).is_dir():
            raise FileNotFoundError(
                f"No diffusion model found at {args.diffusion_model}. Train one with train.py "
                "first, or pass --diffusion-model."
            )
        diffusion = DiffusionModel.from_preset(
            args.diffusion_model, guidance=args.guidance, operator="inpainting"
        )
    if "supervised" in args.methods:
        if args.supervised_model is None:
            raise ValueError('Method "supervised" requires --supervised-model.')
        supervised = UNet.from_preset(args.supervised_model)
        if diffusion is not None:
            assert list(supervised.input_shape) == list(diffusion.input_shape)
    model = diffusion or supervised
    if model is None:  # interpolation only: native resolution, [-1, 1]
        return None, None, None, (-1.0, 1.0)
    return diffusion, supervised, tuple(model.input_shape[:2]), tuple(model.input_range)


def run_diffusion(model, measurements, masks, args: Args, seed):
    """SeqDiff over frames, all B-planes (azimuth) of a frame sampled as one batch."""
    seqdiff = SeqDiff(diffusion_model=model)
    guidance_kwargs = {"omega": args.omega}
    if args.guidance == "dps_smooth_B_plane":
        guidance_kwargs["alpha"] = args.alpha
    out = seqdiff(
        # (frames, azimuth, ...) -> (azimuth, frames, ...): azimuth is the batch axis
        ops.swapaxes(measurements, 0, 1),
        masks=ops.swapaxes(ops.broadcast_to(masks, ops.shape(measurements)), 0, 1),
        num_steps=args.num_steps,
        min_num_steps=args.min_num_steps,
        num_particles=args.num_particles,
        buffer_length=args.buffer_length,
        guidance_kwargs=guidance_kwargs,
        init_guidance_kwargs={**guidance_kwargs, "omega": args.init_omega or args.omega},
        batch_size=args.batch_size,
        seed=seed,
        verbose=args.verbose,
    )
    samples = ops.clip(out["output"], *model.input_range)  # (azimuth, frames, particles, ...)
    timing = seqdiff.time.get_stats("posterior_sample")
    return ops.swapaxes(ops.mean(samples, axis=2), 0, 1), timing


def make_masks(image_shape, n_frames, args: Args):
    """Subsampling masks, with ``buffer_length`` fully observed initial frames."""
    b = args.buffer_length
    if b >= n_frames:
        raise ValueError(
            f"buffer_length ({b}) must be smaller than the number of frames ({n_frames})."
        )
    masks, rate = get_masks(image_shape, n_frames - b, args.subsampling_rate, args.line_width)
    return ops.concatenate([ops.ones((b, *ops.shape(masks)[1:])), masks]), rate


def reconstruct(clean, masks, diffusion, supervised, norm_range, args: Args, seed):
    """Subsample ``clean`` with ``masks`` and reconstruct it with every method in ``args.methods``.

    Returns:
        outputs: ``{method: (frames - buffer_length, azimuth, depth, elevation, 1)}``.
        timing: posterior sampling time stats of the diffusion model (or None).
    """
    b = args.buffer_length
    # Measurements y = A(x) as defined by the inpainting operator (unobserved pixels set to 0).
    # Note that DPS uses the non-squared L2 norm ||y - A(x0)||, so this affects the guidance strength.
    measurements = InpaintingOperator().forward(clean, ops.cast(masks, "bool"))
    outputs, timing = {}, None
    for method in args.methods:
        log.info(f"Running {log.green(method)}...")
        t0 = time.perf_counter()
        if method == "diffusion":
            outputs[method], timing = run_diffusion(diffusion, measurements, masks, args, seed)
        elif method == "supervised":
            outputs[method] = predict_supervised(
                supervised, measurements[b:], masks[b:], args.batch_size
            )
        else:
            outputs[method] = interpolate(
                measurements[b:], masks[b:], order=int(method == "linear")
            )
        log.info(f"{method} took {time.perf_counter() - t0:.1f} s")
    return outputs, timing


def finite_mean(values):
    """Mean over finite values (PSNR is inf for planes that are reconstructed exactly)."""
    values = np.asarray(values)
    return float(np.mean(values[np.isfinite(values)]))


def compute_metrics(outputs, clean, masks, image_range, lpips=True):
    """Image quality on the non-acquired elevation planes only, with ``zea.metrics.Metrics``.

    - B-plane: the B-planes (depth x elevation) with the acquired elevation columns removed.
    - A-plane: the A-planes (depth x azimuth) at the non-acquired elevations.
    """
    missing = ops.convert_to_numpy(masks[:, 0, 0, :, 0]) == 0  # (frames, elevation)
    names = ["mse", "mae", "psnr", "ssim"] + (["lpips"] if lpips else [])
    if missing[0].sum() < SSIM_KWARGS["filter_size"]:
        log.warning("Too few non-acquired elevation planes for the SSIM window, skipping SSIM.")
        names.remove("ssim")
    # Images are passed in [0, 255]: in zea 0.1.9 LPIPS is only correct for that range (zea#715)
    metrics = Metrics(names, image_range=(0, 255), **SSIM_KWARGS)

    def planes(x):
        x = ops.convert_to_numpy(translate(ops.clip(x, *image_range), image_range, (0, 255)))
        b = np.stack([x[t][..., missing[t], :] for t in range(len(x))])  # drop acquired columns
        return b_planes(b), np.asarray(a_planes(x))[missing]

    clean_b, clean_a = planes(clean)
    results = {}
    for name, images in outputs.items():
        images_b, images_a = planes(images)
        kwargs = dict(average_batches=False, mapped_batch_size=64)
        b = metrics(clean_b, images_b, **kwargs)
        a = metrics(clean_a, images_a, **kwargs)
        results[name] = {
            "B_plane": {k: finite_mean(v) for k, v in b.items()},
            "A_plane": {k: finite_mean(v) for k, v in a.items()},
        }
    return results


def print_metrics(results):
    names = list(next(iter(results.values()))["B_plane"])
    header = f"{'':<12}" + "".join(f"{p + '-' + n:>10}" for p in "BA" for n in names)
    lines = [header]
    for method, res in results.items():
        row = "".join(f"{res[p + '_plane'][n]:>10.3f}" for p in "BA" for n in names)
        lines.append(f"{TITLES[method]:<12}" + row)
    log.info("Image quality on non-acquired planes (B: B-planes, A: A-planes)\n" + "\n".join(lines))


def make_figures(outputs, clean, masks, volume, image_range, out_dir: Path, rate):
    """B-plane (center azimuth) and A-plane (a non-acquired elevation) per method, per frame."""
    set_plot_style()
    n_frames, n_az, _, n_el, _ = ops.shape(clean)
    mask_np = ops.convert_to_numpy(ops.broadcast_to(masks, ops.shape(clean)))
    volumes = {TITLES["masks"]: mask_np}
    ordered = sorted(
        outputs, key=list(TITLES).index
    )  # paper order: nearest, linear, U-Net, diffusion
    volumes.update({TITLES[k]: ops.convert_to_numpy(outputs[k]) for k in ordered})
    volumes[TITLES["clean"]] = ops.convert_to_numpy(clean)

    frames = []
    for t in range(n_frames):
        missing = np.flatnonzero(mask_np[t, 0, 0, :, 0] == 0)
        e = int(missing[np.argmin(np.abs(missing - n_el // 2))]) if len(missing) else n_el // 2
        row_b, row_a = {}, {}
        for title, vol in volumes.items():
            is_mask = title == TITLES["masks"]
            vrange = (0, 1) if is_mask else image_range
            b = to_display(vol[t, n_az // 2, ..., 0][None], vrange, volume.grid, "B")[0]
            a = to_display(np.swapaxes(vol[t, :, :, e, 0], 0, 1)[None], vrange, volume.grid, "A")[0]
            row_b[title], row_a[title] = (b / 255, a / 255) if is_mask else (b, a)
        frustum = None
        if volume.grid is not None:  # polar volume: show which elevation planes were acquired
            theta = np.linspace(*volume.grid["theta_range"], n_az)
            phi = np.linspace(*volume.grid["phi_range"], n_el)
            frustum = dict(
                grid=volume.grid,
                acquired_phi=phi[mask_np[t, 0, 0, :, 0] > 0.5],
                b_plane_theta=theta[n_az // 2],
                a_plane_phi=phi[e],
            )
        fig = plot_comparison(
            [row_b, row_a],
            [f"B-plane (a={n_az // 2})", f"A-plane (e={e})"],
            aspect="equal" if volume.grid else "auto",
            frustum=frustum,
            suptitle=f"r = {rate}, frame {t}",
        )
        if t == n_frames - 1:
            fig.savefig(out_dir / "comparison.png", dpi=200, bbox_inches="tight")
            fig.savefig(out_dir / "comparison.pdf", bbox_inches="tight")
        fig.set_dpi(150)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3])
        plt.close(fig)
    if len(frames) > 1:
        imageio.mimsave(out_dir / "comparison.gif", frames, duration=500, loop=0)
    log.success(f"Saved figures to {log.yellow(out_dir)}")


def main(args: Args):
    zea.init_device(args.device)
    seed = jax.random.PRNGKey(args.seed)

    diffusion, supervised, image_shape, norm_range = load_models(args)

    path = resolve_file(args.data, args.file, seed=args.seed)
    log.info(f"Reconstructing {log.yellow(path)}")
    volume = load_volume(path, n_frames=args.num_frames, db_range=args.db_range)
    image_shape = image_shape or tuple(volume.data.shape[2:4])  # (depth, elevation)
    clean = preprocess(volume.data, image_shape, volume.image_range, norm_range)

    masks, rate = make_masks(image_shape, ops.shape(clean)[0], args)
    outputs, timing = reconstruct(clean, masks, diffusion, supervised, norm_range, args, seed)
    b = args.buffer_length

    # metrics and figures exclude the (fully observed) buffer frames
    clean, masks = clean[b:], masks[b:]
    results = compute_metrics(outputs, clean, masks, norm_range, lpips=args.lpips)
    print_metrics(results)
    if timing is not None:
        log.info(
            f"Diffusion: {timing['median']:.3f} s per volume ({1 / timing['median']:.2f} volumes/s)"
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"file": str(path), "effective_subsampling_rate": rate, "metrics": results}
    summary["args"] = {k: str(v) if isinstance(v, Path) else v for k, v in asdict(args).items()}
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    log.success(f"Saved metrics to {log.yellow(out_dir / 'metrics.json')}")
    np.savez_compressed(
        out_dir / "outputs.npz",
        clean=ops.convert_to_numpy(clean),
        masks=ops.convert_to_numpy(masks),
        **{k: ops.convert_to_numpy(v) for k, v in outputs.items()},
    )
    make_figures(outputs, clean, masks, volume, norm_range, out_dir, rate)


if __name__ == "__main__":
    main(tyro.cli(Args))
