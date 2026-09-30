import torch
import torch.nn as nn
from collections import OrderedDict
import einops
from einops.layers.torch import Rearrange
from einops import rearrange
from torch.amp import autocast
from tok_utils import all_gather 

class VectorQuantizer(torch.nn.Module):
    def __init__(self,
                 codebook_size: int = 1024,
                 token_size: int = 256,
                 commitment_cost: float = 0.25,
                 use_l2_norm: bool = False,
                 clustering_vq: bool = False
                 ):
        super().__init__()
        self.codebook_size = codebook_size
        self.token_size = token_size
        self.commitment_cost = commitment_cost

        self.embedding = torch.nn.Embedding(codebook_size, token_size)
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)
        self.use_l2_norm = use_l2_norm

        self.clustering_vq = clustering_vq
        if clustering_vq:
            self.decay = 0.99
            self.register_buffer("embed_prob", torch.zeros(self.codebook_size))
        self.step = 0

    # Ensure quantization is performed using f32
    @autocast("cuda", enabled=False)
    def forward(self, z, clustering_vq=None):
        if clustering_vq == None :
            clustering_vq = self.clustering_vq
        z = z.float()
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = rearrange(z, 'b h w c -> (b h w) c')
        unnormed_z_flattened = z_flattened

        if self.use_l2_norm:
            z_flattened = torch.nn.functional.normalize(z_flattened, dim=-1)
            embedding = torch.nn.functional.normalize(self.embedding.weight, dim=-1)
        else:
            embedding = self.embedding.weight
        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(embedding**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, embedding.T)

        min_encoding_indices = torch.argmin(d, dim=1) # num_ele
        z_quantized = self.get_codebook_entry(min_encoding_indices).view(z.shape).contiguous()

        if self.use_l2_norm:
            z = torch.nn.functional.normalize(z, dim=-1)

        # compute loss for embedding
        commitment_loss = self.commitment_cost * ((z_quantized.detach() - z) **2)
        codebook_loss = (z_quantized - z.detach()) **2 
        if clustering_vq and self.training:
            with torch.no_grad():
                # Gather distance matrix from all GPUs. 
                # encoding_indices = gather(min_encoding_indices) 
                encoding_indices = torch.cat(all_gather(min_encoding_indices), 0)  
                if len(min_encoding_indices.shape) != 1:
                    raise ValueError(f"min_encoding_indices in a wrong shape, {min_encoding_indices.shape}")
                # Compute and update the usage of each entry in the codebook. 
                encodings = torch.zeros(encoding_indices.shape[0], self.codebook_size, device=z.device) 
                encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)  
                avg_probs = torch.mean(encodings, dim=0) 
                self.embed_prob.mul_(self.decay).add_(avg_probs, alpha=1-self.decay) 
                # Closest sampling to update the codebook.
                # all_d = gather(d)
                all_d = torch.cat(all_gather(d), 0) 
                # all_unnormed_z_flattened = gather(unnormed_z_flattened).detach()
                all_unnormed_z_flattened = torch.cat(all_gather(unnormed_z_flattened), 0).detach()
                if all_d.shape[0] != all_unnormed_z_flattened.shape[0]:
                    raise ValueError(
                        "all_d and all_unnormed_z_flattened have different length" + 
                        f"{all_d.shape}, {all_unnormed_z_flattened.shape}")
                indices = torch.argmin(all_d, dim=0) 
                random_feat = all_unnormed_z_flattened[indices]
                # Decay parameter based on the average usage.
                decay = torch.exp(-(self.embed_prob * self.codebook_size * 10) /
                                   (1 - self.decay) - 1e-3).unsqueeze(1).repeat(1, self.token_size)  
                self.embedding.weight.data = self.embedding.weight.data * (1 - decay) + random_feat * decay 
        loss = commitment_loss + codebook_loss

        # preserve gradients
        z_quantized = z + (z_quantized - z).detach()

        # reshape back to match original input shape
        z_quantized = rearrange(z_quantized, 'b h w c -> b c h w').contiguous()
        min_encoding_indices = min_encoding_indices.view(-1).contiguous() 
        min_encoding_indices = torch.cat(all_gather(min_encoding_indices), 0)  
        result_dict = dict(
            quantizer_loss=loss,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
            min_encoding_indices=min_encoding_indices
        ) 
        return z_quantized, result_dict

    def get_codebook_entry(self, indices):
        if len(indices.shape) == 1:
            z_quantized = self.embedding(indices)
        elif len(indices.shape) == 2:
            z_quantized = torch.einsum('bd,dn->bn', indices, self.embedding.weight)
        else:
            raise NotImplementedError
        if self.use_l2_norm:
            z_quantized = torch.nn.functional.normalize(z_quantized, dim=-1)
        return z_quantized

class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model,
            n_head,
            mlp_ratio = 4.0,
            act_layer = nn.GELU,
            norm_layer = nn.LayerNorm, 
        ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = Attention(d_model, num_heads=n_head)
        self.mlp_ratio = mlp_ratio
        # optionally we can disable the FFN
        if mlp_ratio > 0:
            self.ln_2 = norm_layer(d_model)
            mlp_width = int(d_model * mlp_ratio)
            self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, mlp_width)),
                ("gelu", act_layer()),
                ("c_proj", nn.Linear(mlp_width, d_model))
            ]))
 
    def forward(self, x, attn_mask=None, is_causal=False
    ): 
        attn_output = self.attn(self.ln_1(x), attn_mask, is_causal)
        x = x + attn_output
        if self.mlp_ratio > 0:
            x = x + self.mlp(self.ln_2(x))
        return x
    
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., attn_mask=None):
        super().__init__() 
        head_dim = dim // num_heads
        self.num_heads = num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.in_proj = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop) 

    def forward(self, x, attn_mask, is_causal): 
        qkv = self.in_proj(x)
        qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads).float()
        q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D 
        if attn_mask != None:
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask.to(q.device))
        elif is_causal:
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = einops.rearrange(x, 'B H L D -> B L (H D)')
        
        x = self.out_proj(x)
        x = self.proj_drop(x)
        return x

def _expand_token(token, batch_size: int):
    return token.unsqueeze(0).expand(batch_size, -1, -1)


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__() 
        self.image_size = config.img_size
        self.patch_size = config.patch_size
        self.grid_size = self.image_size // self.patch_size 

        self.num_latent_tokens = self.grid_size**2 
        self.token_size = config.token_size 

        self.width = 768
        self.num_layers = 12
        self.num_heads = 12
        
        self.patch_embed = nn.Conv2d(
            in_channels=3, out_channels=self.width,
              kernel_size=self.patch_size, stride=self.patch_size, bias=True)
        
        scale = self.width ** -0.5 
        self.class_embedding = nn.Parameter(scale * torch.randn(1, self.width))
        self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_size**2, self.width))
        self.latent_token_positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens + 1, self.width))
        self.ln_pre = nn.LayerNorm(self.width)
        self.transformer = nn.ModuleList()
        for i in range(self.num_layers):
            self.transformer.append(ResidualAttentionBlock(
                self.width, self.num_heads, mlp_ratio=4.0
            ))
        self.ln_post = nn.LayerNorm(self.width)
        self.conv_out = nn.Conv2d(self.width, self.token_size, kernel_size=1, bias=True)   
        
    def forward(self, pixel_values, latent_tokens): 
        batch_size = pixel_values.shape[0]
        x = pixel_values
        x = self.patch_embed(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1) 
        
        # positional embeddings 
        x = x + self.positional_embedding.to(x.dtype) 
        
        latent_tokens = _expand_token(latent_tokens, x.shape[0]).to(x.dtype)
        latent_tokens = torch.cat([_expand_token(self.class_embedding, latent_tokens.shape[0]).to(latent_tokens.dtype), 
                                   latent_tokens], dim=1)
        latent_tokens = latent_tokens + self.latent_token_positional_embedding.to(x.dtype) 
        
        x = torch.cat([x, latent_tokens], dim=1)

        x = self.ln_pre(x)  
        for i in range(self.num_layers):
            x = self.transformer[i](x)  
        latent_tokens = x[:, -self.num_latent_tokens:]    
        
        latent_tokens = self.ln_post(latent_tokens)
        # fake 2D shape
        latent_tokens = latent_tokens.reshape(batch_size, self.num_latent_tokens, self.width, 1).permute(0, 2, 1, 3)
        latent_tokens = self.conv_out(latent_tokens) 
        latent_tokens = latent_tokens.reshape(batch_size, self.token_size, 1, self.num_latent_tokens) 
        
        return latent_tokens
    

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__() 
        self.image_size = config.img_size
        self.patch_size = config.patch_size
        self.grid_size = self.image_size // self.patch_size 
        self.dynamic_training = config.get("dynamic_training", False)
        self.distillation_training = config.get("distillation_training", False)
        self.num_latent_tokens = self.grid_size**2
        self.token_size = config.token_size 
        self.width = 1024
        self.num_layers = 24
        self.num_heads = 16  

        self.decoder_embed = nn.Linear(
            self.token_size, self.width, bias=True)
        scale = self.width ** -0.5 
        self.class_embedding = nn.Parameter(scale * torch.randn(1, self.width)) 

        # add mask token and query pos embed
        self.mask_token = nn.Parameter(scale * torch.randn(1, 1, self.width))
        self.positional_embedding = nn.Parameter(
        scale * torch.randn(self.grid_size ** 2 + 1, self.width))

        self.latent_token_positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.width))
        self.ln_pre = nn.LayerNorm(self.width)
        self.transformer = nn.ModuleList()
        for i in range(self.num_layers):
            self.transformer.append(ResidualAttentionBlock(
                self.width, self.num_heads, mlp_ratio=4.0 
            ))
        self.ln_post = nn.LayerNorm(self.width)

        # Directly predicting RGB pixels
        self.ffn = nn.Sequential(
            nn.Conv2d(self.width, self.patch_size * self.patch_size * 3, 1, padding=0, bias=True),
            Rearrange('b (p1 p2 c) h w -> b c (h p1) (w p2)',
                p1 = self.patch_size, p2 = self.patch_size),)
        self.conv_out = nn.Conv2d(3, 3, 1, padding=0, bias=True)
    
    def forward(self, z_quantized): 
        N, C, H, W = z_quantized.shape 

        if self.dynamic_training or self.distillation_training:
            assert H == 1 and W <= self.num_latent_tokens, f"{H}, {W}, {self.num_latent_tokens}"
        else:
            # relaxed assertion condition
            assert H == 1 and W == self.num_latent_tokens, f"{H}, {W}, {self.num_latent_tokens}"

        x = z_quantized.reshape(N, C*H, W).permute(0, 2, 1) # NLD
        x = self.decoder_embed(x)

        batchsize, seq_len, _ = x.shape

        mask_tokens = self.mask_token.repeat(batchsize, self.grid_size**2, 1).to(x.dtype)
        mask_tokens = torch.cat([_expand_token(self.class_embedding, mask_tokens.shape[0]).to(mask_tokens.dtype),
                                    mask_tokens], dim=1)
        mask_tokens = mask_tokens + self.positional_embedding.to(mask_tokens.dtype)
        x = x + self.latent_token_positional_embedding[:seq_len]
        x = torch.cat([x, mask_tokens], dim=1)

        
        x = self.ln_pre(x)
        for i in range(self.num_layers):
            x = self.transformer[i](x, is_causal=True)
         

        x = x[:, -self.grid_size**2:] 
        x = self.ln_post(x)

        x = x.permute(0, 2, 1).reshape(batchsize, self.width, self.grid_size, self.grid_size) 
        x = self.ffn(x.contiguous())
        x = self.conv_out(x)
        x = nn.Sigmoid()(x)
        

        
        return x
    
class VAutoencoder(nn.Module): 
    def __init__(
        self, config,
    ):
        super().__init__()

        self.image_size = config.img_size
        self.patch_size = config.patch_size
        self.grid_size = self.image_size // self.patch_size 

        # modules
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        self.dynamic_training = config.get("dynamic_training", False)
        self.distillation_training = config.get("distillation_training", False)

        self.num_latent_tokens = self.grid_size**2
        scale = self.encoder.width ** -0.5
        self.latent_tokens = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.encoder.width))
         
        # VQ
        self.codebook_size = config.codebook_size
        self.commitment_cost = 0.25
        self.token_size = config.token_size
        self.use_l2_norm = False 
        self.clustering_vq = config.clustering_vq
        self.quantize = VectorQuantizer(
                codebook_size=self.codebook_size,
                token_size=self.token_size,
                commitment_cost=self.commitment_cost,
                use_l2_norm=self.use_l2_norm,
                clustering_vq=self.clustering_vq)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def forward(self, x, mode_type="largest"):
        z_quantized, result_dict  = self.encode(x, self.latent_tokens) 
        x_recon = self.decode(z_quantized, mode_type=mode_type) 
        result_dict["quantizer_loss"] = result_dict["quantizer_loss"].mean()
        return x_recon, result_dict, result_dict["min_encoding_indices"]  
    
    def encode(self, x, latent_tokens): 
        x = self.encoder(x, latent_tokens) 
        z_quantized, result_dict = self.quantize(x) 
        return z_quantized, result_dict
    
    def decode(self, x, mode_type="largest"): 
        """
        x: z_quantized, shape: B x token_size x 1 x num_latent_tokens
        """

        if self.dynamic_training or self.distillation_training:
            # MARK: Apply Tail Token Drop in training
            x = self.apply_tail_token_drop(x, mode_type=mode_type)

        x = self.decoder(x)
        return x
    
    def apply_tail_token_drop(self, x: torch.Tensor, mode_type="largest"):
        """
        Apply Tail Token Drop to the tensor.
        """
        # x: B x token_size x 1 x num_latent_tokens
        _, _, _, num_tokens = x.shape
        # take index randomly
        if mode_type == "largest":
            # return largest length tokens
            return x
        elif mode_type == "uniform":
            end_indices = torch.randint(32, num_tokens + 1, (1,)).to(x.device)
            
        elif mode_type == "smallest":
            end_indices = torch.tensor(1).to(x.device)
        # apply tail token drop
        x = x[:, :, :, :end_indices]
        # print(f"##### Tail Token Drop: {end_indices.item()}/{num_tokens}")
        return x



def NesTok(config):
    model = VAutoencoder(config)
    return model
