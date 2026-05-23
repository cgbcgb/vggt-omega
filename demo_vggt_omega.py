import argparse
import os

import cv2
import numpy as np
import open3d as o3d
import torch

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera
from demo_gradio import unproject_depth_map_to_point_map


def parse_args():
    parser = argparse.ArgumentParser(description="VGGT-Omega inference demo")
    parser.add_argument("--checkpoint", required=True, help="Path to VGGT-Omega checkpoint.")
    parser.add_argument("--images", nargs="+", required=True, help="Paths to input images.")
    parser.add_argument("--image-resolution", type=int, default=512, help="Input image resolution. Default: 512.")
    parser.add_argument("--output-dir", default=None, help="Directory to save depth maps.")
    return parser.parse_args()


def main():
    args = parse_args()

    model = VGGTOmega().to("cuda").eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))

    images = load_and_preprocess_images(args.images, image_resolution=args.image_resolution).to("cuda")

    with torch.inference_mode():
        predictions = model(images)

    extrinsics, intrinsics = encoding_to_camera(
        predictions["pose_enc"],
        predictions["images"].shape[-2:],
    )

    depth = predictions["depth"]
    depth_conf = predictions["depth_conf"]
    camera_and_register_tokens = predictions["camera_and_register_tokens"]
    camera_tokens = camera_and_register_tokens[:, :, :1]
    registers = camera_and_register_tokens[:, :, 1:]

    # Save depth maps as colormapped PNGs
    depth_np = depth.squeeze(0).squeeze(-1).cpu().numpy()  # (B, H, W)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    for i in range(depth_np.shape[0]):
        d = depth_np[i]
        d_min, d_max = d.min(), d.max()
        d_norm = ((d - d_min) / (d_max - d_min + 1e-8) * 255).astype(np.uint8)
        d_color = cv2.applyColorMap(d_norm, cv2.COLORMAP_TURBO)
        out_path = os.path.join(output_dir, f"depth_{i:04d}.png")
        cv2.imwrite(out_path, d_color)

    print(f"Saved {depth_np.shape[0]} depth maps to {output_dir}")

    # Reconstruct 3D point cloud with RGB colors
    depth_map_np = depth.squeeze(0).cpu().numpy()  # (B, H, W, 1)
    images_np = predictions["images"].squeeze(0).cpu().numpy()  # (B, 3, H, W)
    images_np = np.transpose(images_np, (0, 2, 3, 1))  # (B, H, W, 3), range [0, 1]
    extrinsics_np = extrinsics.squeeze(0).cpu().numpy()  # (B, 3, 4)
    intrinsics_np = intrinsics.squeeze(0).cpu().numpy()  # (B, 3, 3)

    world_points = unproject_depth_map_to_point_map(depth_map_np, extrinsics_np, intrinsics_np)  # (B, H, W, 3)

    stride = 2
    all_points = world_points[:, ::stride, ::stride].reshape(-1, 3)
    all_colors = images_np[:, ::stride, ::stride].reshape(-1, 3)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors.clip(0, 1))

    pcd_path = os.path.join(output_dir, "pointcloud.pcd")
    o3d.io.write_point_cloud(pcd_path, pcd)
    print(f"Saved point cloud ({all_points.shape[0]} points) to {pcd_path}")

    print(f"Processed {len(args.images)} images.")
    print(f"Output shapes — depth: {depth.shape}, extrinsics: {extrinsics.shape}, intrinsics: {intrinsics.shape}")


if __name__ == "__main__":
    main()
