# NesTok: Nested Self-Aligned 1D Tokenizer for Autoregressive Image Generation

📎 [Nested Self-Aligned 1D Tokenizer for Autoregressive Image Generation]()

Authors: Jiawei Zhang, Shuhao Liu, Rong Huang, Yuancheng Li, Zhihui Li, Xiaojun Chang, Changlin Li

<p align="center">
    <img src="./imgs/visualization_imagenet.jpg" width="800"/> <br />
    <em> 
    </em>
</p>


## Abstract <a name="2"></a> 
One-dimensional (1D) variable-length visual tokenizers enable adaptive compression by varying the number of tokens, allowing downstream autoregressive (AR) models to flexibly trade off generation quality against computational cost using a single tokenizer. However, existing approaches based on nested dropout often fail to fully exploit the representational capacity of the tokenizer, resulting in suboptimal performance in both image reconstruction and generation. In this work, we introduce NesTok, a nested self-alignment framework tailored to dynamic visual tokenizers. NesTok introduces cross-length training, which jointly optimizes reconstruction across token lengths while using the full-length sequence to guide shorter counterparts, enabling shorter token sequences to approach the reconstruction quality of full-length sequences. On ImageNet, NesTok improves substantially over standard training and achieves an rFID score of 0.98. On downstream image generation, it achieves the state-of-the-art gFID score of \textbf{1.46} on ImageNet 256$\times$256 among existing variable-length autoregressive image generation methods.


<p align="center">
    <img src="./imgs/method.jpg" width="800"/> <br />
    <em> 
    </em>
</p>



## ✏️ Usage

### Download Trained Models

The trained checkpoints are hosted on Hugging Face in [Jiawei804/NesTok](https://huggingface.co/Jiawei804/NesTok). Place the downloaded checkpoints in the local `weights/` directory, preserving the filenames.

| AR Model | Checkpoint | FID | IS | #Params |
|-----------|---------|---------|-------|---------|
| NesTok-B | [NesTok-b.bin](https://huggingface.co/Jiawei804/NesTok/resolve/main/NesTok-b.bin) | 1.73 | 250.5 | 177M |
| NesTok-L | [NesTok-l.bin](https://huggingface.co/Jiawei804/NesTok/resolve/main/NesTok-l.bin) | 1.50 | 278.9 | 318M |
| NesTok-XL | [NesTok-xl.bin](https://huggingface.co/Jiawei804/NesTok/resolve/main/NesTok-xl.bin) | 1.46 | 295.9 | 662M |

All AR models use the tokenizer checkpoint [NesTok](https://huggingface.co/Jiawei804/NesTok/resolve/main/NesTok.pth).

Install the Hugging Face CLI, log in with an account that has access to the repository, and download the checkpoints:

```bash
python -m pip install --upgrade huggingface_hub
hf auth login
hf download Jiawei804/NesTok NesTok.pth NesTok-b.bin NesTok-l.bin NesTok-xl.bin --local-dir weights
```

To evaluate only NesTok-B, download `NesTok.pth` and `NesTok-b.bin`.



### Generation (ImageNet 256x256)

You can prepare [ADM evaluation](https://github.com/openai/guided-diffusion) script first.

Sampling with NesTok-B w/ cfg:
```bash  
torchrun --nnodes=1 --nproc_per_node=8 sample_imagenet.py \
  --tokenizer_ckpt weights/NesTok.pth \
  --config configs/NesTok_b.yaml \
  --experiment_output_dir samples/NesTok_b \
  --experiment_generator_checkpoint weights/NesTok-b.bin
```

If you want to evaluate gFID w/o cfg, you can set `guidance_scale=0` and `randomize_temperature=0.95` in `configs/*.yaml` files.

### Reconstruction (ImageNet 256x256)
Reconstruct the images with NesTok:
```bash  
torchrun --nnodes=1 --nproc_per_node=8 --master_port 12345 recon_for_evaluation.py \
	--data_path="your_path/ImageNet-1K/val" \
	--batch_size 32 \
	--ckpt_path="weights/NesTok.pth" \
	--config="configs/NesTok_b.yaml" \  # all setting is the same
	--token_len 32 \

```

```bash 
python	calculate_metrics.py \
	--recon_dir "./samples/recon_samples" \
	--orig_dir "./samples/real_samples" \
```

### Training Tokenizer
Please refer to [train_tokenizer/README_NesTok.md](train_tokenizer/README_NesTok.md).

### Training Autoregressive Models
Pretokenize the dataset
```bash  
torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 pretokenization.py \
--img_size 256 \
--batch_size 32 \
--ten_crop \
--data_path ${PATH_TO_IMAGENET} \
```

Reproduce NesTok-B (400 epochs, total 250k steps)
```bash  
accelerate launch --num_machines=1 --num_processes=8 train_ar.py \
    config=configs/NesTok_b.yaml  \
    experiment.output_dir="results/NesTok_b" \
```

## Acknowledgements
A large portion of codes in this repo is based on [One-D-Piece](https://github.com/turingmotors/One-D-Piece), [AliTok](https://github.com/ali-vilab/alitok), [MAR](https://github.com/LTH14/mar). We are grateful for these open-source research projects.


## Citation
If you use our code for your paper, please cite:

```bibtex

```

## License
This project is under the MIT license. See [LICENSE](LICENSE.txt) for details.
