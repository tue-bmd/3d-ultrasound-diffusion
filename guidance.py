"""DPS guidance with a total-variation smoothness prior across neighbouring B-planes.

The diffusion prior is trained on 2D B-planes (depth x elevation). A volume is reconstructed by
sampling all B-planes (one per azimuth angle) as a batch. Independently sampled B-planes can be
inconsistent with each other, so we add a total-variation (TV) penalty along the batch axis,
i.e. across *azimuth*, to the DPS measurement error.

Importing this module registers the guidance under the name ``"dps_smooth_B_plane"``.
"""

from keras import ops
from zea.func import L2
from zea.internal.registry import diffusion_guidance_registry
from zea.models.diffusion import DPS


def tv_norm(x, axis=0, epsilon=1e-8):
    """(Smoothed) total variation of ``x`` along ``axis``: sum(sqrt(diff(x)^2 + eps))."""
    diff = ops.diff(x, axis=axis)
    return ops.sum(ops.sqrt(diff**2 + epsilon))


@diffusion_guidance_registry(name="dps_smooth_B_plane")
class DPSSmoothBPlane(DPS):
    """DPS with an additional TV prior across the batch (azimuth) axis.

    Note:
        The batch must contain *neighbouring* B-planes of a single volume and a single particle,
        in azimuth order. When sampling in chunks (``batch_size``) the TV term only couples
        B-planes within the same chunk.
    """

    def compute_error(
        self,
        noisy_images,
        measurements,
        noise_rates,
        signal_rates,
        omega: float,
        alpha: float = 0.0,
        **kwargs,
    ):
        """Measurement error ``omega * ||y - A(x0)||^2 + alpha * TV_azimuth(x0)``.

        Args:
            noisy_images: Noisy B-planes ``x_t``, shape ``(n_azimuth, depth, elevation, 1)``.
            measurements: Observed (masked) B-planes ``y``.
            noise_rates: Noise rates at the current diffusion time.
            signal_rates: Signal rates at the current diffusion time.
            omega: Weight of the measurement (data consistency) error.
            alpha: Weight of the TV smoothness prior across B-planes.
            **kwargs: Forwarded to the operator (e.g. ``mask``).

        Returns:
            ``(error, (pred_noises, pred_images))``
        """
        pred_noises, pred_images = self.diffusion_model.denoise(
            noisy_images, noise_rates, signal_rates, training=False
        )
        measurement_error = omega * L2(measurements - self.operator.forward(pred_images, **kwargs))
        smoothness = tv_norm(pred_images, axis=0)
        return measurement_error + alpha * smoothness, (pred_noises, pred_images)
