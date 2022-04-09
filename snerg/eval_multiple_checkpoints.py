# coding=utf-8
# Lint as: python3
"""Evaluation script for Nerf."""
import csv
import functools
from os import path

from absl import app
from absl import flags
from flask import Flask
import flax
from flax.metrics import tensorboard
from flax.training import checkpoints
import jax
from jax import random
import numpy as np
import tensorflow as tf
import tensorflow_hub as tf_hub

from snerg.model_zoo import datasets
from snerg import model_zoo
from snerg.model_zoo import utils

FLAGS = flags.FLAGS

utils.define_flags()


def main(unused_argv):
    # Hide the GPUs and TPUs from TF so it does not reserve memory on them for
    # LPIPS computation or dataset loading.
    tf.config.experimental.set_visible_devices([], "GPU")
    tf.config.experimental.set_visible_devices([], "TPU")

    rng = random.PRNGKey(20200823)

    if FLAGS.config is not None:
        utils.update_flags(FLAGS)
    if FLAGS.train_dir is None:
        raise ValueError("train_dir must be set. None set now.")
    if FLAGS.data_dir is None:
        raise ValueError("data_dir must be set. None set now.")

    dataset = datasets.get_dataset("test", FLAGS)
    rng, key = random.split(rng)
    model, init_variables = model_zoo.get_model(key, dataset.peek(), FLAGS)
    optimizer = flax.optim.Adam(FLAGS.lr_init).create(init_variables)
    state = utils.TrainState(optimizer=optimizer)
    del optimizer, init_variables

    # Rendering is forced to be deterministic even if training was randomized, as
    # this eliminates "speckle" artifacts.
    def render_fn(variables, key_0, key_1, rays):
        return jax.lax.all_gather(
            model.apply(variables, key_0, key_1, rays, False), axis_name="batch"
        )

    # pmap over only the data input.
    render_pfn = jax.pmap(
        render_fn,
        in_axes=(None, None, None, 0),
        donate_argnums=3,
        axis_name="batch",
    )

    # Compiling to the CPU because it's faster and more accurate.
    ssim_fn = jax.jit(functools.partial(utils.compute_ssim, max_val=1.0), backend="cpu")

    last_step = 0
    out_dir = path.join(
        FLAGS.train_dir, "path_renders" if FLAGS.render_path else "test_preds"
    )
    if not FLAGS.eval_once:
        summary_writer = tensorboard.SummaryWriter(path.join(FLAGS.train_dir, "eval"))
    # CSV: save the results of the model in tabular form
    checkpoint_metrics = list()
    # unlike the provided eval.py, we will eval on ALL the checkpoints
    first_checkpoint = 0 + FLAGS.save_every
    last_checkpoint = FLAGS.max_steps + FLAGS.save_every
    for step in range(first_checkpoint, last_checkpoint, FLAGS.save_every):
        try:
            state = checkpoints.restore_checkpoint(FLAGS.train_dir, state, step=step)
        except:  # early out in case the checkpoint isn't available
            print(f"Could not locate checkpoint_{step}")
            break
        if step <= last_step:
            print(f"Exiting because step > last_step: {(step, last_step)}")
            continue
        if FLAGS.save_output and (not utils.isdir(out_dir)):
            utils.makedirs(out_dir)
        psnr_values = []
        ssim_values = []
        showcase_index = None
        if not FLAGS.eval_once:
            showcase_index = np.random.randint(0, dataset.size)
        # CSV: before eval'ing samples - init the "row" this represents in our CSV
        csv_row = {"Checkpoint": step}
        # Let's only evalulate on about 10% of the test dataset
        step_size = dataset.size // 10
        need_showcase = False
        for idx in range(0, dataset.size, step_size):
            print(f"Evaluating {idx+1}/{dataset.size}")
            batch = next(dataset)
            (
                pred_color,
                pred_disp,
                pred_acc,
                pred_features,
                pred_specular,
            ) = utils.render_image(
                functools.partial(render_pfn, state.optimizer.target),
                batch["rays"],
                rng,
                FLAGS.dataset == "llff",
                chunk=FLAGS.chunk,
            )
            if jax.host_id() != 0:  # Only record via host 0.
                continue

            if not FLAGS.eval_once and idx == showcase_index:
                showcase_color = pred_color
                showcase_disp = pred_disp
                showcase_acc = pred_acc
                showcase_features = pred_features
                showcase_specular = pred_specular
                if not FLAGS.render_path:
                    showcase_gt = batch["pixels"]
                need_showcase = True

            if not FLAGS.render_path:
                psnr = utils.compute_psnr(((pred_color - batch["pixels"]) ** 2).mean())
                ssim = ssim_fn(pred_color, batch["pixels"])
                print(f"PSNR = {psnr:.4f}, SSIM = {ssim:.4f}")
                psnr_values.append(float(psnr))
                ssim_values.append(float(ssim))
                # CSV: save the PSNR on this sample
                csv_row[f"Sample_{idx + 1}_PSNR"] = f"{psnr:.4f}"

            if FLAGS.save_output:
                utils.save_img(pred_color, path.join(out_dir, "{:03d}.png".format(idx)))
                utils.save_img(
                    pred_disp[Ellipsis, 0],
                    path.join(out_dir, "disp_{:03d}.png".format(idx)),
                )
        optim_step = int(state.optimizer.state.step)
        if need_showcase and (not FLAGS.eval_once) and (jax.host_id() == 0):
            summary_writer.image("pred_color", showcase_color, optim_step)
            summary_writer.image("pred_disp", showcase_disp, optim_step)
            summary_writer.image("pred_acc", showcase_acc, optim_step)
            summary_writer.image("pred_features", showcase_features, optim_step)
            summary_writer.image("pred_specular", showcase_specular, optim_step)
            if not FLAGS.render_path:
                summary_writer.scalar(
                    "psnr", np.mean(np.array(psnr_values)), optim_step
                )
                summary_writer.scalar(
                    "ssim", np.mean(np.array(ssim_values)), optim_step
                )
                summary_writer.image("target", showcase_gt, step)

        if FLAGS.save_output and (not FLAGS.render_path) and (jax.host_id() == 0):
            with utils.open_file(
                path.join(out_dir, f"psnrs_{optim_step}.txt"), "w"
            ) as f:
                f.write(" ".join([str(v) for v in psnr_values]))
            with utils.open_file(
                path.join(out_dir, f"ssims_{optim_step}.txt"), "w"
            ) as f:
                f.write(" ".join([str(v) for v in ssim_values]))
            with utils.open_file(path.join(out_dir, "psnr.txt"), "w") as f:
                f.write("{}".format(np.mean(np.array(psnr_values))))
            with utils.open_file(path.join(out_dir, "ssim.txt"), "w") as f:
                f.write("{}".format(np.mean(np.array(ssim_values))))

        # CSV: before next checkpoint, save the current row
        checkpoint_metrics.append(csv_row)

        # if FLAGS.eval_once:
        #     break
        # if int(step) >= FLAGS.max_steps:
        #     break
        last_step = step
    # Save to CSV, if needed (credit to nikhilaggarwal3 for this code: https://www.geeksforgeeks.org/writing-csv-files-in-python/)
    if FLAGS.csv_output_filename:
        with open(FLAGS.csv_output_filename, "w") as csvfile:
            # creating a csv dict writer object
            writer = csv.DictWriter(csvfile, fieldnames=FLAGS.csv_output_headers)
            writer.writeheader()  # write the headers
            writer.writerows(checkpoint_metrics)  # write the data rows


if __name__ == "__main__":
    app.run(main)
