# coding=utf-8
# Lint as: python3

from typing import Any, Callable

from flax import linen as nn
from jax import random
import jax.numpy as jnp
import torch

from snerg.model_zoo import utils
from TensoRF.models import tensoRF
import TensoRF.utils as trf_utils


class TensorfModel(nn.Module):
    """TensoRF NN Model."""

    num_coarse_samples: int  # The number of samples for the coarse nerf.
    num_fine_samples: int  # The number of samples for the fine nerf.
    use_viewdirs: bool  # If True, use viewdirs as an input.
    near: float  # The distance to the near plane
    far: float  # The distance to the far plane
    noise_std: float  # The std dev of noise added to raw sigma.
    net_depth: int  # The depth of the first part of MLP.
    net_width: int  # The width of the first part of MLP.
    num_viewdir_channels: int  # The number of extra channels for view-dependence.
    viewdir_net_depth: int  # The depth of the view-dependence MLP.
    viewdir_net_width: int  # The width of the view-dependence MLP.
    net_activation: Callable[..., Any]  # MLP activation
    skip_layer: int  # How often to add skip connections.
    num_rgb_channels: int  # The number of RGB channels.
    num_sigma_channels: int  # The number of density channels.
    white_bkgd: bool  # If True, use a white background.
    min_deg_point: int  # The minimum degree of positional encoding for positions.
    max_deg_point: int  # The maximum degree of positional encoding for positions.
    deg_view: int  # The degree of positional encoding for viewdirs.
    lindisp: bool  # If True, sample linearly in disparity rather than in depth.
    rgb_activation: Callable[..., Any]  # Output RGB activation.
    sigma_activation: Callable[..., Any]  # Output sigma activation.
    legacy_posenc_order: bool  # Keep the same ordering as the original tf code.
    decomp_method: str  # The method of decomposition. 1 of ['CP', 'VM', 'VMSplit'].
    checkpoint_path: str  # The checkpoint saved from training (using the TensoRF modules).
    is_llff_360_scene: bool
    dataset: str  # the type of data loader used, e.g. "blender", "llff", etc.
    N_voxel_init: int  # controls the resolution of matrix and vector used in decomposition
    num_ray_samples: int  # the number of sampling points on each ray

    DECOMP_METHODS = {
        "CP": tensoRF.TensorCP,
        "VM": tensoRF.TensorVM,
        "VMSplit": tensoRF.TensorVMSplit,
    }

    DEVICE_BACKEND = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup(self):
        # A: grab the checkpoint file
        ckpt = torch.load(self.checkpoint_path, map_location=self.DEVICE_BACKEND)
        kwargs = ckpt["kwargs"]
        kwargs.update({"device": self.DEVICE_BACKEND})
        # B: instantiate the appropiate TensoRF
        self.tensorf = self.DECOMP_METHODS[self.decomp_method](**kwargs)
        self.tensorf.load(ckpt)

    def determine_num_samples(self) -> int:
        """Uses the config file and other info to give a precise answer."""
        # A: initialize all the variables needed to find the number of samples
        scene_bbox_sizes = {
            # these values are from the TensoRF.dataLoader package
            "blender": torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]]),
            "llff": torch.tensor([[-1.5, -1.67, -1.0], [1.5, 1.67, 1.0]]),
        }
        aabb = scene_bbox_sizes[self.dataset_type]
        reso_cur = trf_utils.N_to_reso(self.N_voxel_init, aabb)
        # B: now we can calculate the num_samples!
        true_num_samples = min(
            self.num_ray_samples,
            trf_utils.cal_n_samples(reso_cur, self.tensorf.step_ratio),
        )
        return true_num_samples

    @nn.compact
    def __call__(self, rng_0, rng_1, rays, randomized):
        """Returns the predictions of TensoRF.

        Parameters:
            rng_0: jnp.ndarray, random number generator for coarse model sampling.
            rng_1: jnp.ndarray, random number generator for fine model sampling.
            rays: util.Rays, a namedtuple of ray origins, directions, and viewdirs.
                  Assumed to be just 1 batch.
            randomized: bool, use randomized stratified sampling.

        Returns:
            tuple, (rgb, sigma, alpha, weight, bg_weight)
        """
        # make prediction
        num_samples = self.determine_num_samples()
        rgb_map, depth_map = self.tensorf(
            rays,
            is_train=False,  # train using the PyTorch classes
            white_bg=self.white_bkgd,
            ndc_ray=self.is_llff_360_scene,
            N_samples=num_samples,
        )
        # we're not called directly during baking - return as is
        return rgb_map, depth_map  # rgb, sigma, alpha, weight, bg_weight


def construct_tensorf(key, example_batch, args):
    """Construct a JAX-compatible, Tensorial Radiance Field.

    Parameters:
        key: jnp.ndarray. Random number generator.
        example_batch: dict, an example of a batch of data.
        args: FLAGS class. Hyperparameters of nerf.

    Returns:
        model: nn.Model. TensoRF model with parameters.
        state: flax.Module.state. odel state for stateful parameters.
    """
    net_activation = nn.relu
    rgb_activation = nn.sigmoid
    sigma_activation = nn.relu

    # Assert that rgb_activation always produces outputs in [0, 1], and
    # sigma_activation always produce non-negative outputs.
    x = jnp.exp(jnp.linspace(-90, 90, 1024))
    x = jnp.concatenate([-x[::-1], x], 0)

    rgb = rgb_activation(x)
    if jnp.any(rgb < 0) or jnp.any(rgb > 1):
        raise NotImplementedError(
            "Choice of rgb_activation `{}` produces colors outside of [0, 1]".format(
                args.rgb_activation
            )
        )

    sigma = sigma_activation(x)
    if jnp.any(sigma < 0):
        raise NotImplementedError(
            "Choice of sigma_activation `{}` produces negative densities".format(
                args.sigma_activation
            )
        )

    # init the model
    model = TensorfModel(
        min_deg_point=args.min_deg_point,
        max_deg_point=args.max_deg_point,
        deg_view=args.deg_view,
        num_coarse_samples=args.num_coarse_samples,
        num_fine_samples=args.num_fine_samples,
        use_viewdirs=args.use_viewdirs,
        near=args.near,
        far=args.far,
        noise_std=args.noise_std,
        white_bkgd=args.white_bkgd,
        net_depth=args.net_depth,
        net_width=args.net_width,
        num_viewdir_channels=args.num_viewdir_channels,
        viewdir_net_depth=args.viewdir_net_depth,
        viewdir_net_width=args.viewdir_net_width,
        skip_layer=args.skip_layer,
        num_rgb_channels=args.num_rgb_channels,
        num_sigma_channels=args.num_sigma_channels,
        lindisp=args.lindisp,
        net_activation=net_activation,
        rgb_activation=rgb_activation,
        sigma_activation=sigma_activation,
        legacy_posenc_order=args.legacy_posenc_order,
        decomp_method=args.tensorf_method,
        checkpoint_path=args.tensorf_checkpoint,
        is_llff_360_scene=(args.dataset == "llff" and not args.spherify),
        dataset_type=args.dataset,
        N_voxel_init=args.N_voxel_init,
        num_ray_samples=args.num_ray_samples,
    )
    rays = example_batch["rays"]
    key1, key2, key3 = random.split(key, num=3)

    init_variables = model.init(
        key1,
        rng_0=key2,
        rng_1=key3,
        rays=utils.namedtuple_map(lambda x: x[0], rays),
        randomized=args.randomized,
    )

    return model, init_variables
