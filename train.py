"""Train the 2D B-plane diffusion prior.

The prior models single B-planes (depth x elevation). Every B-plane (one per azimuth angle) of
every frame of the training volumes is a training sample, loaded with ``zea.Dataloader`` from a
local folder or the Hugging Face Hub.

Example:
    python train.py --data /path/to/data/train

The trained model is saved as a zea preset in ``--output-dir``, which is also where
``inference.py`` loads the diffusion model from by default.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("MPLBACKEND", "Agg")  # figures are only saved, never shown

import csv
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import keras
import matplotlib.pyplot as plt
import numpy as np
import tyro
import yaml
import zea
from keras import ops
from zea import log
from zea.models.diffusion import DiffusionModel

import paths
from data import VALUES, list_files, value_range


@dataclass
class Args:
    data: str = str(paths.TRAIN_DIR)
    """Local folder with hdf5 files, or ``hf://<repo>/<subdir>``."""
    output_dir: Path = paths.TRAINED_DIFFUSION_DIR
    init_from: Optional[str] = None
    """Optional preset (local folder or ``hf://<repo>``) to fine-tune from."""

    image_shape: Tuple[int, int] = (400, 48)
    """B-plane shape (depth, elevation) the data is resized to."""
    widths: List[int] = field(default_factory=lambda: [32, 64, 96, 128])
    block_depth: int = 2
    min_signal_rate: float = 0.02
    max_signal_rate: float = 0.95

    epochs: int = 100
    steps_per_epoch: Optional[int] = None
    """Training steps per epoch (default: one pass over all B-planes)."""
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    ema: float = 0.999
    db_range: Tuple[float, float] = (-60.0, 0.0)
    """Dynamic range for images stored in dB (float32)."""
    cache: bool = False
    """Keep the loaded data in RAM (faster for datasets on a network drive)."""
    seed: int = 42
    device: str = "auto:1"


def make_dataloader(args: Args) -> zea.Dataloader:
    """Shuffled batches of single B-planes ``(batch, depth, elevation, 1)`` in [-1, 1]."""
    files = list_files(args.data)
    if not files:
        raise FileNotFoundError(f"No hdf5 files found in {args.data}")
    with zea.File(files[0], "r") as f:
        image_range = value_range(f.dataset(VALUES).dtype, args.db_range)
    loader = zea.Dataloader(
        args.data,
        key=VALUES,  # (frames, depth, azimuth, elevation)
        additional_axes_iter=(2,),  # one sample per azimuth angle, i.e. per B-plane
        batch_size=args.batch_size,
        image_size=args.image_shape,
        image_range=image_range,
        normalization_range=(-1, 1),
        clip_image_range=True,
        drop_remainder=True,
        shuffle=True,
        cache=args.cache,
        seed=args.seed,
    )
    log.info(f"Training on {len(loader) * args.batch_size} B-planes from {len(files)} files.")
    return loader


def batches(loader):
    """Endless iterator over the (reshuffled every pass) dataloader."""
    while True:
        yield from loader


def build_model(args: Args) -> DiffusionModel:
    if args.init_from is not None:
        log.info(f"Initializing from {log.yellow(args.init_from)}")
        return DiffusionModel.from_preset(args.init_from, guidance=None, operator=None)
    return DiffusionModel(
        input_shape=(*args.image_shape, 1),
        input_range=(-1, 1),
        min_signal_rate=args.min_signal_rate,
        max_signal_rate=args.max_signal_rate,
        network_kwargs={"widths": args.widths, "block_depth": args.block_depth},
        ema_val=args.ema,
        guidance=None,
        operator=None,
    )


def make_train_step(model: DiffusionModel, optimizer, ema: float):
    """Jitted, functional (stateless) training step for the JAX backend."""
    network = model.network

    def loss_fn(trainable, non_trainable, images, key):
        key_noise, key_time = jax.random.split(key)
        noises = jax.random.normal(key_noise, images.shape)
        times = jax.random.uniform(
            key_time, (images.shape[0], 1, 1, 1), minval=model.min_t, maxval=model.max_t
        )
        noise_rates, signal_rates = model.diffusion_schedule(times)
        noisy_images = signal_rates * images + noise_rates * noises
        pred_noises, non_trainable = network.stateless_call(
            trainable, non_trainable, [noisy_images, noise_rates**2], training=True
        )
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        noise_loss = jnp.mean((pred_noises - noises) ** 2)
        image_loss = jnp.mean(jnp.abs(pred_images - images))
        return noise_loss, (non_trainable, image_loss)

    @jax.jit
    def train_step(state, images, key):
        trainable, non_trainable, opt_vars, ema_trainable = state
        (noise_loss, (non_trainable, image_loss)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(trainable, non_trainable, images, key)
        trainable, opt_vars = optimizer.stateless_apply(opt_vars, grads, trainable)
        ema_trainable = [ema * e + (1 - ema) * w for e, w in zip(ema_trainable, trainable)]
        state = (trainable, non_trainable, opt_vars, ema_trainable)
        return state, {"noise_loss": noise_loss, "image_loss": image_loss}

    return train_step


def ema_variables(model, variables):
    """EMA-network counterparts of ``model.network`` variables (matched as in zea's train_step)."""
    pairs = {id(w): e for w, e in zip(model.network.weights, model.ema_network.weights)}
    return [pairs[id(v)] for v in variables]


def write_state(model, optimizer, state):
    """Copy the functional training state back into the keras variables."""
    trainable, non_trainable, opt_vars, ema_trainable = state
    network = model.network
    for var, value in zip(network.trainable_variables, trainable):
        var.assign(value)
    for var, value in zip(network.non_trainable_variables, non_trainable):
        var.assign(value)
    for var, value in zip(optimizer.variables, opt_vars):
        var.assign(value)
    for var, value in zip(ema_variables(model, network.trainable_variables), ema_trainable):
        var.assign(value)
    for var, value in zip(ema_variables(model, network.non_trainable_variables), non_trainable):
        var.assign(value)


def save_samples(model, path, seed, n=8):
    """Unconditional samples from the (EMA) model."""
    samples = ops.convert_to_numpy(model.sample(n_samples=n, n_steps=50, seed=seed))
    fig, axes = plt.subplots(1, n, figsize=(1.2 * n, 4))
    for ax, image in zip(axes, samples[..., 0]):
        ax.imshow(image, cmap="gray", vmin=-1, vmax=1, aspect="auto")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main(args: Args):
    zea.init_device(args.device)
    keras.utils.set_random_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_args.yaml").write_text(
        yaml.safe_dump({k: str(v) if isinstance(v, Path) else v for k, v in asdict(args).items()})
    )

    dataloader = make_dataloader(args)
    steps_per_epoch = args.steps_per_epoch or len(dataloader)
    loader = batches(dataloader)
    model = build_model(args)
    optimizer = keras.optimizers.AdamW(args.learning_rate, weight_decay=args.weight_decay)
    optimizer.build(model.network.trainable_variables)
    train_step = make_train_step(model, optimizer, args.ema)

    state = (
        [v.value for v in model.network.trainable_variables],
        [v.value for v in model.network.non_trainable_variables],
        [v.value for v in optimizer.variables],
        # EMA starts from the online weights (the cloned EMA network has its own random init)
        [v.value for v in model.network.trainable_variables],
    )
    key = jax.random.PRNGKey(args.seed)
    log_file = out_dir / "log.csv"
    with open(log_file, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "noise_loss", "image_loss", "seconds"])

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        progbar = keras.utils.Progbar(steps_per_epoch, unit_name="step")
        losses = []
        for step in range(steps_per_epoch):
            key, step_key = jax.random.split(key)
            images = jnp.asarray(next(loader), dtype=jnp.float32)
            state, logs = train_step(state, images, step_key)
            losses.append([float(logs["noise_loss"]), float(logs["image_loss"])])
            progbar.update(step + 1, list(zip(("n_loss", "i_loss"), losses[-1])))
        noise_loss, image_loss = np.mean(losses, axis=0)
        with open(log_file, "a", newline="") as f:
            csv.writer(f).writerow([epoch, noise_loss, image_loss, time.perf_counter() - t0])

        write_state(model, optimizer, state)
        model.save_to_preset(str(out_dir))
        save_samples(model, out_dir / f"samples_epoch_{epoch:03d}.png", jax.random.PRNGKey(epoch))
        log.success(f"Epoch {epoch}: n_loss={noise_loss:.4f}, saved to {log.yellow(out_dir)}")

    loader.close()  # stop the (possibly mid-pass) iterator and release file handles
    dataloader.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
