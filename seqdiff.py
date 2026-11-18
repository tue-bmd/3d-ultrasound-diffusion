"""SeqDiff: sequential diffusion posterior sampling for (ultrasound) sequences.

Every frame is reconstructed with diffusion posterior sampling. Instead of starting from pure
noise, frame ``t`` starts from the reconstruction of frame ``t-1``, noised to an intermediate
diffusion step, so only ``min_num_steps`` of the ``num_steps`` trajectory are needed per frame.

Reference:
    T. S. W. Stevens, O. Nolan, J.-L. Robert, R. J. G. van Sloun, "Sequential Posterior Sampling
    with Diffusion Models", IEEE International Conference on Acoustics, Speech and Signal
    Processing (ICASSP), 2025. https://arxiv.org/abs/2409.05399,
    doi:10.1109/ICASSP49660.2025.10889752
"""

import time
from typing import Optional

import jax
import keras
from keras import ops
from zea import log
from zea.func import vmap
from zea.models.diffusion import DiffusionModel
from zea.utils import FunctionTimer


class SeqDiff:
    """Sequential diffusion posterior sampling (SeqDiff), Stevens et al., ICASSP 2025.

    See "Sequential Posterior Sampling with Diffusion Models",
    https://arxiv.org/abs/2409.05399.
    """

    def __init__(self, diffusion_model: DiffusionModel):
        """
        Args:
            diffusion_model: diffusion model with guidance and operator set.
        """
        self.diffusion_model = diffusion_model
        # time every posterior sampling call (one call per frame)
        self.time = FunctionTimer()
        self.diffusion_model.posterior_sample = self.time(self.diffusion_model.posterior_sample)

    def __call__(
        self,
        measurements,
        masks,
        guidance_kwargs: dict,
        num_steps: int = 100,
        min_num_steps: int = 4,
        num_particles: int = 1,
        buffer_length: int = 0,
        init_guidance_kwargs: Optional[dict] = None,
        stochastic_sampling: bool = False,
        batch_size: Optional[int] = None,
        seed=None,
        verbose: bool = False,
    ) -> dict:
        """Reconstruct a batch of sequences.

        Args:
            measurements: ``(batch, frames, height, width, channels)``.
            masks: same shape as ``measurements``, 1 where measured.
            guidance_kwargs: kwargs for the guidance function (e.g. ``omega``, ``alpha``).
            num_steps: number of steps of the full diffusion trajectory.
            min_num_steps: number of steps run per frame when warm-starting from the
                previous frame. ``min_num_steps == num_steps`` means every frame starts from
                noise (vanilla diffusion posterior sampling).
            num_particles: number of posterior samples per measurement.
            buffer_length: the first ``buffer_length`` frames are assumed fully observed and
                only used to initialize the sequence (not reconstructed).
            init_guidance_kwargs: guidance kwargs for a first frame that starts from noise
                (when ``buffer_length == 0``). Defaults to ``guidance_kwargs``.
            stochastic_sampling: DDPM instead of DDIM sampling.
            batch_size: number of images sampled in parallel (None: whole batch at once).
            seed: jax PRNG key. Every frame and every sampled chunk gets its own key from it.
            verbose: show progress.

        Returns:
            dict with ``"output"`` of shape ``(batch, frames - buffer_length, particles, h, w, c)``.
        """
        measurements = ops.convert_to_tensor(measurements)
        assert ops.ndim(measurements) == 5, (
            f"measurements must be (batch, frames, h, w, c), got {measurements.shape}"
        )
        assert ops.ndim(masks) == 5, f"masks must be (batch, frames, h, w, c), got {masks.shape}"
        batch, total_frames = ops.shape(measurements)[:2]
        assert 0 <= buffer_length < total_frames, (
            f"buffer_length must be in [0, {total_frames}), got {buffer_length}"
        )
        seed = jax.random.PRNGKey(0) if seed is None else seed
        initial_step = num_steps - min_num_steps
        init_guidance_kwargs = init_guidance_kwargs or guidance_kwargs

        # (batch, frames, ...) -> (particles * batch, frames, ...), particle-major so that
        # neighbouring batch elements stay neighbouring B-planes (needed for the TV prior)
        masks = ops.broadcast_to(masks, ops.shape(measurements))
        measurements = ops.tile(measurements, (num_particles, 1, 1, 1, 1))
        masks = ops.tile(masks, (num_particles, 1, 1, 1, 1))

        if buffer_length > 0 and initial_step > 0:
            # warm start from the last (fully observed) buffer frame
            previous = measurements[:, buffer_length - 1]
            log.info(f"SeqDiff: initializing with {buffer_length} fully sampled frame(s).")
        else:
            previous = None

        mode = "SeqDiff" if initial_step > 0 else "Vanilla diffusion"
        sampler = "DDPM" if stochastic_sampling else "DDIM"
        guidance = type(self.diffusion_model.guidance_fn).__name__
        log.info(f"Running {mode} with {sampler} posterior sampling ({log.green(guidance)})...")

        outputs = []
        progbar = keras.utils.Progbar(
            total_frames - buffer_length, unit_name="frame", verbose=verbose
        )
        for frame_idx in range(buffer_length, total_frames):
            t0 = time.perf_counter()
            if previous is None:
                # starts from noise: first frame without buffer, or vanilla diffusion
                step, kwargs = 0, (init_guidance_kwargs if initial_step > 0 else guidance_kwargs)
            else:
                step, kwargs = initial_step, guidance_kwargs

            def posterior_sample(y, mask, keys, initial_samples=None, _step=step, _kwargs=kwargs):
                # one posterior sample per measurement in the (chunked) batch, own key per chunk
                return self.diffusion_model.posterior_sample(
                    y,
                    n_samples=ops.shape(y)[0],
                    n_steps=num_steps,
                    initial_step=_step,
                    initial_samples=initial_samples,
                    stochastic_sampling=stochastic_sampling,
                    seed=keys[0],
                    mask=mask,
                    **_kwargs,
                )

            keys = jax.random.split(jax.random.fold_in(seed, frame_idx), ops.shape(measurements)[0])
            inputs = (measurements[:, frame_idx], masks[:, frame_idx], keys, previous)
            if batch_size is None or ops.shape(measurements)[0] <= batch_size:
                pred = posterior_sample(*inputs)
            else:
                pred = vmap(posterior_sample, batch_size=batch_size, fn_supports_batch=True)(
                    *inputs
                )

            outputs.append(pred)
            previous = pred if initial_step > 0 else None
            progbar.update(frame_idx - buffer_length + 1)
            if verbose:
                log.info(f"Frame {frame_idx}: {time.perf_counter() - t0:.3f} s")

        output = ops.stack(outputs, axis=1)  # (particles * batch, frames, ...)
        output = ops.reshape(output, (num_particles, batch, *ops.shape(output)[1:]))
        return {"output": ops.moveaxis(output, 0, 2)}  # (batch, frames, particles, ...)
