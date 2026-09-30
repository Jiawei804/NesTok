import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchmetrics.functional.image import (
    peak_signal_noise_ratio as psnr,
    structural_similarity_index_measure as ssim
)
import torch_fidelity


IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".ppm", ".pgm", ".tif", ".tiff", ".webp"}


def _collect_image_files(root_dir):
    files = {}
    root_dir = Path(root_dir)
    for path in sorted(root_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            key = path.relative_to(root_dir).with_suffix("").as_posix()
            if key in files:
                raise ValueError(f"Duplicate image key found under {root_dir}: {key}")
            files[key] = path
    return files


def _load_image_tensor(image_path):
    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
        array = np.asarray(rgb_image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def calculate_psnr_ssim(real_dir, rec_dir, batch_size=256, device="cuda"):
    real_files = _collect_image_files(real_dir)
    rec_files = _collect_image_files(rec_dir)

    missing_recon = sorted(set(real_files) - set(rec_files))
    missing_real = sorted(set(rec_files) - set(real_files))
    if missing_recon or missing_real:
        raise ValueError("Image pairs must match by relative path.")

    keys = sorted(real_files)
    if len(keys) == 0:
        raise ValueError("No matched image pairs found.")

    psnr_list = []
    ssim_list = []
    details = []

    for i in range(0, len(keys), batch_size):
        batch_keys = keys[i:i + batch_size]

        real_batch = []
        rec_batch = []

        for key in batch_keys:
            real_tensor = _load_image_tensor(real_files[key])
            rec_tensor = _load_image_tensor(rec_files[key])

            if real_tensor.shape != rec_tensor.shape:
                raise ValueError(
                    f"Shape mismatch for {key}: real {tuple(real_tensor.shape)}, recon {tuple(rec_tensor.shape)}"
                )

            real_batch.append(real_tensor)
            rec_batch.append(rec_tensor)

        real_batch = torch.stack(real_batch, dim=0).to(device)
        rec_batch = torch.stack(rec_batch, dim=0).to(device)

        with torch.no_grad():
            psnr_value = psnr(rec_batch, real_batch, data_range=1.0)
            ssim_value = ssim(rec_batch, real_batch, data_range=1.0)

        psnr_list.append(psnr_value.cpu())
        ssim_list.append(ssim_value.cpu())


    return {
        "num_pairs": len(keys),
        "psnr": float(torch.stack(psnr_list).mean().item()),
        "ssim": float(torch.stack(ssim_list).mean().item()),
    }


def get_fid_stats(real_dir, rec_dir, fid_stats):
    """Calculate FID statistics between real and reconstructed images."""
    stats = torch_fidelity.calculate_metrics(
        input1=rec_dir,
        input2=real_dir,
        # fid_statistics_file=fid_stats,
        feature_extractor_weights_path="/public/home/cs029/models/Inception/weights-inception-2015-12-05-6726825d.pth",
        cuda=True,
        isc=True,
        fid=True,
        kid=False,
        prc=False,
        verbose=False,
    )
    return stats
fid_stats = "./adm_in256_stats.npz"

def main(args):
    print(get_fid_stats(str(args.orig_dir), str(args.recon_dir), fid_stats))
    # print(calculate_psnr_ssim(str(args.orig_dir), str(args.recon_dir)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recon_dir", type=str, help="Directory containing reconstructed images")
    parser.add_argument("--orig_dir", type=str, help="Directory containing original images")
    # parser.add_argument("--metrics", nargs="+", help="List of metrics to calculate")
    # parser.add_argument("--save_path", type=str, help="Path to save metrics")
    args = parser.parse_args()
    main(args)