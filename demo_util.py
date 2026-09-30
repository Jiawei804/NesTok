""" This file is borrowed from:
    https://github.com/bytedance/1d-tokenizer/blob/main/demo_util.py
"""
import torch
from omegaconf import OmegaConf
from modeling.ar import ARModel


def get_config_cli():
    cli_conf = OmegaConf.from_cli()

    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)

    return conf

def get_config(config_path):
    conf = OmegaConf.load(config_path)
    return conf


def get_ar_generator(config):
    model_cls = ARModel
    generator = model_cls(config)
    generator.load_state_dict(torch.load(config.experiment.generator_checkpoint, map_location="cpu"))
    generator.eval()
    generator.requires_grad_(False) 
    return generator


@torch.no_grad()
def sample_fn(generator,
              tokenizer,
              labels=None,
              guidance_scale=3.0,
              guidance_decay="constant",
              guidance_scale_pow=3.0,
              randomize_temperature=2.0,
              softmax_temperature_annealing=False,
              num_sample_steps=8,
              device="cuda",
              return_tensor=False):
    generator.eval()
    tokenizer.eval()
    if labels is None:
        # macaw, beacon, otter, Siberian husky, peacock, balloon, volcano, lion, golden retriever, random
        labels = [88, 437, 360, 250, 387, 417, 980, 291, 207, torch.randint(0, 999, size=(1,))]

    if not isinstance(labels, torch.Tensor):
        labels = torch.LongTensor(labels).to(device)

    generated_tokens = generator.generate(
        condition=labels,
        guidance_scale=guidance_scale,
        guidance_decay=guidance_decay,
        guidance_scale_pow=guidance_scale_pow,
        randomize_temperature=randomize_temperature,
        softmax_temperature_annealing=softmax_temperature_annealing,
        num_sample_steps=num_sample_steps)
    # print("### Generated tokens shape:", generated_tokens.shape)  [10, 256]

    generated_image = tokenizer.decode_tokens(
        generated_tokens.view(generated_tokens.shape[0], -1)
    )
    # generated_image = (generated_image+1)/2
    if return_tensor:
        return generated_image
    
    generated_image = torch.clamp(generated_image, 0.0, 1.0)
    generated_image = (generated_image * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

    return generated_image, generated_tokens
