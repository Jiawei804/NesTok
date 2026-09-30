import argparse
import os
from pathlib import Path
import datetime

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from omegaconf import OmegaConf
import torch_fidelity


class ImageFolderWithPath(ImageFolder):
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        path, _ = self.samples[index]
        return image, label, path, index


def create_npz_from_sample_folder(sample_dir, npz_path, num_images):
    samples = []
    for i in tqdm(range(num_images), desc=f"Building .npz file from {sample_dir} samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num_images, samples.shape[1], samples.shape[2], 3)
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


@torch.no_grad()
def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("DDP requires CUDA devices, but CUDA is not available.")
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError("This script requires torchrun distributed launch (missing RANK/WORLD_SIZE).")

    timeout = datetime.timedelta(days=365)
    dist.init_process_group("nccl", timeout=timeout)
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    world_size = dist.get_world_size()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.set_grad_enabled(False)

    if not os.path.isdir(args.data_path):
        raise FileNotFoundError(f"ImageNet directory not found: {args.data_path}")
    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")
    config = OmegaConf.load(args.config)

    transform = transforms.Compose(
        [
            transforms.Resize(args.img_size),
            # transforms.Resize((args.img_size, args.img_size)),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
        ]
    )

    dataset = ImageFolderWithPath(args.data_path, transform=transform)
    if len(dataset) == 0:
        raise ValueError(f"No images found in: {args.data_path}")

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    output_dir = Path(args.output_dir)
    rec_dir = output_dir / "recon_samples"
    real_dir = output_dir / "real_samples"
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        rec_dir.mkdir(parents=True, exist_ok=True)
        real_dir.mkdir(parents=True, exist_ok=True)
        print(f"Starting rank={rank}, world_size={world_size}, device={device}")
        print(f"Total number of images that will be reconstructed: {len(dataset)}")
    dist.barrier()

    print("Using NesTok")
    # NesTok or Nested_Dropout using the same model
    from modeling.nestok import NesTok
    model = NesTok(config)
        
    checkpoint = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    if "ema_state" in checkpoint and args.use_ema:
        checkpoint = checkpoint["ema_state"]
        print("use ema_state!!!")
    elif "model_state" in checkpoint:
        checkpoint = checkpoint["model_state"]
        print("use model_state!!!")

    info = model.load_state_dict(checkpoint, strict=False)
    print(f"unused keys: {info[1]}")
    print(f"args.token_len: {args.token_len}")
    model.to(device)
    model.eval()


    if rank == 0:
        print(f"###Total parameters: {sum(p.numel() for p in model.parameters())}")
        print(f"ckpt: {args.ckpt_path}")

    pbar = tqdm(loader, total=len(loader), desc="Reconstructing ImageNet val") if rank == 0 else loader
    processed_local = 0
    for images, _labels, paths, global_indices in pbar:
        images = images.to(device, non_blocking=True)
        recon_images, _, _ = model(images, latent_tokens_len=args.token_len)

        # # sample for stabilityai/sd-vae-ft-mse
        # x = model.encode(images).latent_dist.sample()
        # recon_images = model.decode(x).sample

        recon_images = torch.clamp(recon_images, 0.0, 1.0)
        recon_images = (recon_images * 255.0).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        bsz = recon_images.shape[0]
        # images = (images * 255.0).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

        for j in range(bsz):
            global_idx = int(global_indices[j])
            Image.fromarray(recon_images[j]).save(rec_dir / f"{global_idx:06d}.png")
            Image.fromarray(images[j]).save(real_dir / f"{global_idx:06d}.png") # once is ok !
            

        processed_local += bsz
        if rank == 0:
            pbar.set_postfix({"local_done": processed_local, "world": world_size})


    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reconstruct ImageNet-1K val images with NesTok and export NPZ for rFID"
    )
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--output_dir", type=str, default="samples/")
    parser.add_argument("--ckpt_path", type=str, default="weights/NesTok.pth")
    parser.add_argument("--img_size", type=int, default=256, help="Image size for reconstruction")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for reconstruction")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of dataloader workers")
    parser.add_argument("--config", type=str, default="configs/DTok.yaml", help="Path to DTok config")
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA weights for reconstruction")
    parser.add_argument("--token_len", type=int, default=256, help="Number of latent tokens to keep during reconstruction")
    args = parser.parse_args()

    main(args)
