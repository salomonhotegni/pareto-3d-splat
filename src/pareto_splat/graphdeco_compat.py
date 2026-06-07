"""Compatibility fixes for the pinned GraphDeCo baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def composite_rgba_image(
    image: Image.Image,
    *,
    white_background: bool,
) -> Image.Image:
    """Composite an image's alpha channel onto the configured solid background."""

    rgba_image = image.convert("RGBA")
    channel_value = 255 if white_background else 0
    background = Image.new(
        "RGBA",
        rgba_image.size,
        (channel_value, channel_value, channel_value, 255),
    )
    return Image.alpha_composite(background, rgba_image).convert("RGB")


def install_nerf_synthetic_compositing_patch() -> None:
    """Patch the pinned loader so NeRF Synthetic RGBA images become RGB."""

    import cv2
    from scene.cameras import Camera
    import utils.camera_utils as camera_utils

    original_load_cam = camera_utils.loadCam

    def load_cam(
        args: Any,
        camera_id: int,
        cam_info: Any,
        resolution_scale: float,
        is_nerf_synthetic: bool,
        is_test_dataset: bool,
    ) -> Any:
        if not is_nerf_synthetic:
            return original_load_cam(
                args,
                camera_id,
                cam_info,
                resolution_scale,
                is_nerf_synthetic,
                is_test_dataset,
            )

        image = composite_rgba_image(
            Image.open(cam_info.image_path),
            white_background=args.white_background,
        )

        if cam_info.depth_path:
            depth_path = Path(cam_info.depth_path)
            if not depth_path.is_file():
                raise FileNotFoundError(f"depth image not found: {depth_path}")
            invdepthmap = cv2.imread(str(depth_path), -1)
            if invdepthmap is None:
                raise OSError(f"could not decode depth image: {depth_path}")
            invdepthmap = invdepthmap.astype(np.float32) / 512
        else:
            invdepthmap = None

        original_width, original_height = image.size
        if args.resolution in (1, 2, 4, 8):
            resolution = (
                round(original_width / (resolution_scale * args.resolution)),
                round(original_height / (resolution_scale * args.resolution)),
            )
        else:
            if args.resolution == -1:
                global_downscale = max(original_width / 1600, 1)
            else:
                global_downscale = original_width / args.resolution
            scale = float(global_downscale) * float(resolution_scale)
            resolution = (
                int(original_width / scale),
                int(original_height / scale),
            )

        return Camera(
            resolution,
            colmap_id=cam_info.uid,
            R=cam_info.R,
            T=cam_info.T,
            FoVx=cam_info.FovX,
            FoVy=cam_info.FovY,
            depth_params=cam_info.depth_params,
            image=image,
            invdepthmap=invdepthmap,
            image_name=cam_info.image_name,
            uid=camera_id,
            data_device=args.data_device,
            train_test_exp=args.train_test_exp,
            is_test_dataset=is_test_dataset,
            is_test_view=cam_info.is_test,
        )

    camera_utils.loadCam = load_cam
