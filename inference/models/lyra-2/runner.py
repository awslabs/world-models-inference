# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nvidia Lyra 2.0 — async image-to-3D-world video generation.

In-process load-once runner. Upstream ships inference as a script
(`lyra2_zoomgs_inference.py`) with no packaged pipeline class, so we import
that module — installed as a pinned dependency, see Dockerfile — and reuse its
model loader plus its module-level helpers (`_da3_infer_depth_intrinsics_single`,
`_generate_one_direction`) rather than shelling out to the CLI. Weights load
once in setup(); generate() runs one image through the same zoom-in/zoom-out
flow the script's per-image loop performs.

Single-GPU: context-parallel is not wired, so deploy on one GPU
(WORLD_SIZE=1). On any non-zero rank generate() is a no-op.

Upstream: https://github.com/nv-tlabs/lyra  ·  weights: nvidia/Lyra-2.0
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import tempfile
import shutil
import sys
from datetime import datetime
from pathlib import Path

import torch

from lib.runner import Runner

logger = logging.getLogger(__name__)

DEFAULT_EXPERIMENT = "lyra2"
DEFAULT_PROMPT = "A slow cinematic camera fly-through of the scene."
DEFAULT_RESOLUTION = "480,832"  # H,W
DEFAULT_NUM_FRAMES_ZOOM_IN = 81
DEFAULT_NUM_FRAMES_ZOOM_OUT = 241
DEFAULT_ZOOM_IN_STRENGTH = 0.5
DEFAULT_ZOOM_OUT_STRENGTH = 1.5
DEFAULT_FPS = 16
DEFAULT_SEED = 1


def create_runner() -> "LyraRunner":
    return LyraRunner()


class LyraRunner(Runner):
    """Lyra 2.0 wrapped as an async batch Runner (in-process, load-once)."""

    def setup(self, ckpt_dir: str, device: torch.device, rank: int, world_size: int) -> None:
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.ckpt_dir = ckpt_dir

        if world_size > 1 and rank == 0:
            logger.warning(
                "Lyra runner is single-GPU (WORLD_SIZE=%d); only rank 0 runs inference.",
                world_size,
            )
        if rank != 0:
            return

        # Upstream is run from its repo root (PYTHONPATH=. python -m ...), and it
        # reads config + checkpoints via paths relative to that root. Put the root
        # on sys.path so `import lyra_2` resolves, and chdir so those relative
        # paths (config_file, LoRA, negative_prompt.pt) resolve too.
        self.repo_root = os.environ.get("LYRA_REPO", "/opt/ml/code/lyra/Lyra-2")
        if self.repo_root not in sys.path:
            sys.path.insert(0, self.repo_root)
        os.chdir(self.repo_root)

        from lyra_2._src.inference import lyra2_zoomgs_inference as zg
        from lyra_2._src.utils.model_loader import load_model_from_checkpoint
        from lyra_2._src.inference.depth_utils import load_da3_model

        self.zg = zg  # module: holds the per-image helpers we call in generate()

        # Weights are mounted at ckpt_dir (/opt/ml/model) and follow the HF repo
        # layout: <ckpt_dir>/checkpoints/{model,text_encoder,lora,...}. The upstream
        # scripts assume a checkpoints/ dir relative to cwd; we anchor to ckpt_dir
        # instead so paths resolve regardless of where weights are mounted.
        ckpt_root = os.path.join(ckpt_dir, "checkpoints")
        if not os.path.isdir(ckpt_root):
            # Fall back to ckpt_dir itself if weights aren't nested under checkpoints/.
            ckpt_root = ckpt_dir
        self.ckpt_root = ckpt_root
        model_dir = os.path.join(ckpt_root, "model")

        # Some upstream components (notably the WanVAE) load via paths hardcoded
        # relative to cwd — e.g. "./checkpoints/vae/vae.pth". chdir alone doesn't
        # help since the weights live under ckpt_dir, not the repo. Symlink the
        # repo's expected checkpoints/ dir at the mounted one so every relative
        # "checkpoints/..." reference resolves.
        repo_ckpt_link = os.path.join(self.repo_root, "checkpoints")
        if os.path.abspath(ckpt_root) != os.path.abspath(repo_ckpt_link):
            try:
                if os.path.islink(repo_ckpt_link) or os.path.exists(repo_ckpt_link):
                    if os.path.islink(repo_ckpt_link):
                        os.unlink(repo_ckpt_link)
                if not os.path.exists(repo_ckpt_link):
                    os.symlink(ckpt_root, repo_ckpt_link)
            except OSError as e:
                logger.warning("Could not link %s -> %s: %s", repo_ckpt_link, ckpt_root, e)

        # The diffusion net's runtime dtype is bf16 (model.tensor_kwargs), but
        # build_net materializes it on CUDA in the default dtype *before* weights
        # load. On fp32 default that transient fp32 net is ~44GB and OOMs a 46GB
        # L40S; materializing directly in bf16 (its real dtype) halves that peak
        # and fits comfortably. Restore the default afterwards — DA3 and the
        # sampler math below expect fp32 as the process default.
        prev_default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            logger.info("Loading Lyra 2.0 from %s ...", model_dir)
            self.model, self.config = load_model_from_checkpoint(
                config_file="lyra_2/_src/configs/config.py",
                experiment_name=DEFAULT_EXPERIMENT,
                checkpoint_path=model_dir,
                enable_fsdp=False,
                instantiate_ema=False,
                load_ema_to_reg=False,
                experiment_opts=[],
            )
        finally:
            torch.set_default_dtype(prev_default_dtype)
        dtype = self.model.tensor_kwargs.get("dtype", None)
        dev = self.model.tensor_kwargs.get("device", None)
        if dtype is not None:
            self.model.net = self.model.net.to(device=dev, dtype=dtype)
        self.model.eval()

        # Aux-model device placement. The main diffusion net fills ~34GB of a
        # 46GB L40S at load and climbs to ~45GB during sampling, leaving no room
        # on its GPU for the two auxiliary models each job needs: DA3 (~6GB) and
        # the UMT5-XXL text encoder (~10GB, built on first generate()). When extra
        # GPUs exist we fan those out onto spares so nothing contends with the main
        # net; on a single bigger GPU (e.g. H100 80GB) they all co-locate. Their
        # outputs are small tensors we move back to the main device before use.
        n_gpus = torch.cuda.device_count()
        main_dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
        self.da3_device = "cuda:1" if n_gpus > 1 else main_dev
        self.t5_device = "cuda:2" if n_gpus > 2 else ("cuda:1" if n_gpus > 1 else main_dev)

        # DA3 (depth+intrinsics). Upstream's default is the fine-tuned checkpoint
        # staged alongside the weights (checkpoints/recon/model.pt); prefer it so
        # we don't fetch DA3 from HuggingFace at runtime on the GPU box. Fall back
        # to the pretrained HF repo only if the local checkpoint isn't present.
        da3_ckpt = os.path.join(self.ckpt_root, "recon", "model.pt")
        da3_ckpt = da3_ckpt if os.path.isfile(da3_ckpt) else None
        self.da3_model = load_da3_model(
            da3_model_name="depth-anything/DA3NESTED-GIANT-LARGE-1.1",
            da3_model_path_custom=da3_ckpt,
            device=self.da3_device,
        )
        self.da3_model.eval()

        # Negative-prompt embeddings — loaded once, reused for every job. This
        # file holds only tensors, so load with weights_only=True to block the
        # arbitrary-code-execution path in pickle deserialization.
        self.negative_prompt_data = torch.load(  # nosemgrep
            os.path.join(self.ckpt_root, "text_encoder", "negative_prompt.pt"),
            map_location="cpu", weights_only=True,
        )
        logger.info("Lyra 2.0 ready.")

    # -----------------------------------------------------------------
    def generate(self, **params) -> str:
        """Run one image → combined zoom-out+zoom-in fly-through video."""
        if self.rank != 0:
            return ""

        import cv2
        from lyra_2._ext.imaginaire.utils import misc
        from lyra_2._ext.imaginaire.visualize.video import save_img_or_video
        from lyra_2._src.inference.get_t5_emb import get_umt5_embedding

        image_path = params.get("image_path") or params.get("image")
        if not image_path or not Path(image_path).exists():
            raise ValueError("Lyra requires an input image ('image_path').")
        caption = params.get("prompt") or DEFAULT_PROMPT

        args = self._build_args(params)
        misc.set_random_seed(seed=args.seed, by_rank=True)

        # --- depth + intrinsics (upstream helper) ------------------------
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise ValueError(f"Cannot read image: {image_path}")
        rgb_t = torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        target_h, target_w = (int(x) for x in args.resolution.split(","))
        image_chw01, depth_hw, K_33, mask_hw = self.zg._da3_infer_depth_intrinsics_single(
            da3_model=self.da3_model, img_rgb_uint8=rgb_t, target_hw=(target_h, target_w),
        )

        dev = self.model.tensor_kwargs.get("device", None)
        dtype = self.model.tensor_kwargs.get("dtype", None)
        img_bchw = image_chw01.to(device=dev) * 2.0 - 1.0  # [-1, 1]

        # UMT5-XXL is a ~10GB text encoder built (and cached) on first call. Run
        # it on its own device (a spare GPU when available) so it doesn't contend
        # with the main net on cuda:0, then move the small embedding to the main
        # device for the diffusion pass.
        t5 = get_umt5_embedding(caption, device=self.t5_device).to(device=dev, dtype=dtype)
        if t5.dim() == 2:
            t5 = t5.unsqueeze(0)
        elif t5.dim() == 3 and t5.shape[0] != 1:
            t5 = t5[:1]
        neg_t5 = misc.to(self.negative_prompt_data["t5_text_embeddings"], **self.model.tensor_kwargs)

        # --- generate both directions (upstream helper) ------------------
        common = dict(
            model=self.model, args=args, img_bchw=img_bchw, depth_hw=depth_hw,
            mask_hw=mask_hw, K_33=K_33, t5_embeddings=t5, neg_t5_embeddings=neg_t5,
            da3_model=self.da3_model, process_group=None,
        )
        result_in = self.zg._generate_one_direction(
            trajectory="horizontal_zoom", direction="right",
            strength=args.zoom_in_strength, N=args.num_frames_zoom_in,
            log_prefix="zoom_in", **common,
        )
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        result_out = self.zg._generate_one_direction(
            trajectory="horizontal_zoom", direction="left",
            strength=args.zoom_out_strength, N=args.num_frames_zoom_out,
            log_prefix="zoom_out", **common,
        )
        if result_in is None and result_out is None:
            raise RuntimeError("Lyra produced no frames (both directions failed).")

        # --- combine (zoom-out reversed + zoom-in) and save --------------
        clips = []
        if result_out is not None:
            clips.append(result_out["video"].flip(dims=[2]))
        if result_in is not None:
            clips.append(result_in["video"])
        combined = torch.cat(clips, dim=2)
        combined_01 = (combined[0].clamp(-1, 1) * 0.5 + 0.5).float().cpu()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        stem = str(Path(tempfile.gettempdir()) / "lyra-2" / ts / "output")
        Path(stem).parent.mkdir(parents=True, exist_ok=True)
        save_img_or_video(combined_01, stem, fps=args.fps)  # writes <stem>.mp4
        produced = Path(stem + ".mp4")
        if not produced.exists():
            raise RuntimeError(f"Lyra saved no file at {produced}")

        output_path = params.get("output_path")
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(produced), output_path)
            logger.info("Lyra output → %s", output_path)
            return output_path
        return str(produced)

    # -----------------------------------------------------------------
    def _build_args(self, params) -> argparse.Namespace:
        """Build the args Namespace the upstream sampler reads.

        The AR sampler reads dozens of fields off `args` (guidance, shift,
        num_sampling_step, da3_*, offload*, ...). Rather than hand-list them and
        chase one AttributeError at a time, we reuse upstream's own
        parse_arguments() with a minimal argv — that yields a Namespace with
        EVERY field defaulted exactly as upstream intends — then override only
        the runtime knobs we expose. Two defaults are forced off deliberately:
        use_moge_scale (we don't load the optional MoGe model) and offload*
        (H100 has the headroom; CPU offload just adds latency).

        Version coupling: this relies on upstream's parse_arguments() defaults,
        so it is pinned to the upstream commit built into the image —
        LYRA_SHA=87f79a52b81b366d1d4aa3a526aa12e54207c998 (see Dockerfile). If
        that SHA is bumped, re-verify the defaults below (esp. num_sampling_step,
        guidance, da3_*) haven't drifted.
        """
        argv_backup = sys.argv
        try:
            # --input_image_path is the one required arg; pass a placeholder (we
            # read the image ourselves in generate(), so its value is unused).
            sys.argv = ["lyra2_zoomgs_inference.py", "--input_image_path",
                        str(Path(tempfile.gettempdir()) / "unused.png")]
            args = self.zg.parse_arguments()
        finally:
            sys.argv = argv_backup

        # Runtime knobs from the request.
        args.experiment = params.get("experiment", DEFAULT_EXPERIMENT)
        args.output_path = str(Path(tempfile.gettempdir()) / "lyra-2" / "work")
        args.resolution = params.get("resolution", DEFAULT_RESOLUTION)
        args.seed = int(params.get("seed", DEFAULT_SEED))
        args.fps = int(params.get("fps", DEFAULT_FPS))
        args.num_frames = int(params.get("num_frames", DEFAULT_NUM_FRAMES_ZOOM_IN))
        args.num_frames_zoom_in = int(params.get("num_frames_zoom_in", DEFAULT_NUM_FRAMES_ZOOM_IN))
        args.num_frames_zoom_out = int(params.get("num_frames_zoom_out", DEFAULT_NUM_FRAMES_ZOOM_OUT))
        args.zoom_in_strength = float(params.get("zoom_in_strength", DEFAULT_ZOOM_IN_STRENGTH))
        args.zoom_out_strength = float(params.get("zoom_out_strength", DEFAULT_ZOOM_OUT_STRENGTH))
        args.use_dmd = bool(params.get("use_dmd", False))
        # Diffusion sampling steps. The UI/route send this as `sampling_steps`;
        # upstream names the arg `num_sampling_step`. Only override when supplied
        # so the un-set case keeps upstream's default. Note: _apply_dmd_defaults()
        # below forces this to 4 when use_dmd — DMD's 4-step schedule wins, by design.
        if params.get("sampling_steps") is not None:
            args.num_sampling_step = int(params.get("sampling_steps"))
        args.context_parallel_size = 1
        args.num_samples = 1
        # Forced off (see docstring): no MoGe model loaded; no CPU offload on H100.
        args.use_moge_scale = False
        args.offload = False
        args.offload_when_prompt = False
        args.offload_da3_diffusion = False

        # DMD fast path (4-step sampler, ~15× faster). Delegate to upstream's own
        # helper so the DMD LoRA (checkpoints/lora/dmd_distillation.safetensors)
        # is injected into lora_paths and the DMD scheduler is switched on — just
        # setting use_dmd_scheduler skips the LoRA and produces degraded output.
        args.use_dmd_scheduler = False
        self.zg._apply_dmd_defaults(args)
        return args
