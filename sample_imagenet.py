""" This file is borrowed from:
    https://github.com/bytedance/1d-tokenizer/blob/main/sample_imagenet_rar.py
"""
import demo_util
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
import os 
from tqdm import tqdm 


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def main(args):
    config = demo_util.get_config(args.config)
    config.experiment.output_dir = args.experiment_output_dir
    config.experiment.generator_checkpoint = args.experiment_generator_checkpoint

    num_fid_samples = 10000
    per_proc_batch_size = 125
    sample_folder_dir = config.experiment.output_dir
    seed = config.model.generator.seed 

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.set_grad_enabled(False)

    # setup DDP.
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size() 
    device = rank % torch.cuda.device_count()
    seed = seed + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.") 
    
    dist.barrier()

    generator = demo_util.get_ar_generator(config)

    print("Using NesTok")
    # NesTok or Nested_Dropout using the same model
    from modeling.nestok import NesTok
    tokenizer = NesTok(config)

    checkpoint = torch.load(args.tokenizer_ckpt, map_location='cpu', weights_only=False)  
    if "ema_state" in checkpoint and args.use_ema:
        checkpoint = checkpoint["ema_state"]
        print("use ema_state!!!")
    elif "model_state" in checkpoint:
        checkpoint = checkpoint["model_state"]
        print("use model_state!!!")
    
    info = tokenizer.load_state_dict(checkpoint, strict=False) 
    print(f"unused keys: {info[1]}")
    del checkpoint  
    tokenizer.to(device) 
    generator.to(device)

    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}, image_seq_len: {config.model.generator.image_seq_len}")
        print(f"experiment_generator_checkpoint: {config.experiment.generator_checkpoint}")
        # 打印参数数量
        num_params = sum(p.numel() for p in generator.parameters())
        print(f"Number of parameters in generator: {num_params}")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    assert num_fid_samples % global_batch_size == 0
    if rank == 0:
        print(f"Total number of images that will be sampled: {num_fid_samples}")

    samples_needed_this_gpu = int(num_fid_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0

    all_classes = list(range(config.model.generator.condition_num_classes)) * (num_fid_samples // config.model.generator.condition_num_classes)
    # # this 
    # 88, 437, 360, 250, 387, 417, 980, 291, 207, torch.randint(0, 1000, size=(1,)).item()
    # classes = np.array([387], dtype=np.int64)
    # all_classes = np.repeat(classes, num_fid_samples)

    subset_len = len(all_classes) // world_size
    all_classes = np.array(all_classes[rank * subset_len: (rank+1)*subset_len], dtype=np.int64)
    cur_idx = 0

    for _ in pbar:
        y = torch.from_numpy(all_classes[cur_idx * n: (cur_idx+1)*n]).to(device)
        cur_idx += 1

        samples, generated_tokens = demo_util.sample_fn(
            generator=generator,
            tokenizer=tokenizer,
            labels=y.long(),
            randomize_temperature=config.model.generator.randomize_temperature,
            guidance_scale=config.model.generator.guidance_scale,
            guidance_scale_pow=config.model.generator.guidance_scale_pow,
            device=device
        )
        # Save samples to disk as individual .png files
        for i, sample in enumerate(samples):
            index = i * dist.get_world_size() + rank + total
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        total += global_batch_size

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder_dir, num_fid_samples)
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    import argparse
    def get_args_parser():
        parser = argparse.ArgumentParser('sample config', add_help=False)
        # new
        parser.add_argument("--tokenizer_ckpt", default='weights/NesTok.pth', help="path to tokenizer checkpoint")
        parser.add_argument("--config", default='configs/NesTok-xl.yaml', help="path to config file")
        parser.add_argument("--token_len", default=256, type=int, help="length of latent tokens")
        parser.add_argument("--use_ema", action="store_true", help="whether to use EMA weights for reconstruction")
        parser.add_argument("--experiment_output_dir", default='output/nestok_xl', help="path to output directory")
        parser.add_argument("--experiment_generator_checkpoint", default='weights/NesTok_xl.bin', help="path to generator checkpoint")
        return parser
    
    args = get_args_parser()
    args = args.parse_args()
    main(args)