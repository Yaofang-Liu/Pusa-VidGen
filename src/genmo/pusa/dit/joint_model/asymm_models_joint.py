import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.attention import sdpa_kernel

# import genmo.mochi_preview.dit.joint_model.context_parallel as cp
from genmo.mochi_preview.dit.joint_model.layers import (
    FeedForward,
    PatchEmbed,
    RMSNorm,
    TimestepEmbedder,
)
from genmo.mochi_preview.dit.joint_model.mod_rmsnorm import modulated_rmsnorm
from genmo.mochi_preview.dit.joint_model.residual_tanh_gated_rmsnorm import (
    residual_tanh_gated_rmsnorm,
)
from genmo.mochi_preview.dit.joint_model.rope_mixed import (
    compute_mixed_rotation,
    create_position_matrix,
)
from genmo.mochi_preview.dit.joint_model.temporal_rope import apply_rotary_emb_qk_real
from genmo.mochi_preview.dit.joint_model.utils import (
    AttentionPool,
    modulate,
    pad_and_split_xy,
    unify_streams,
)
import ipdb

COMPILE_FINAL_LAYER = os.environ.get("COMPILE_DIT") == "1"
COMPILE_MMDIT_BLOCK = os.environ.get("COMPILE_DIT") == "1"

from genmo.lib.attn_imports import comfy_attn, flash_varlen_qkvpacked_attn, sage_attn, sdpa_attn_ctx

def all_to_all_collect_heads(x: torch.Tensor) -> torch.Tensor:
    # Make tensor contiguous before view
    return x.contiguous().view(x.size(0), x.size(1), x.size(2) * x.size(3))

def all_to_all_collect_tokens(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    # Move QKV dimension to the front.
    #   B M (3 H d) -> 3 B M H d
    B, M, _ = x.size()
    x = x.contiguous().view(B, M, 3, num_heads, -1)
    return x.permute(2, 0, 1, 3, 4).contiguous()


# TODO Only linear, initialize the FrameMixer module to act as an identity function
class FrameMixer(nn.Module):
    def __init__(self, F, num_layers=2, expansion=4, bias=True, device=None):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            # Create sequential layers
            seq = nn.Sequential(
                nn.Linear(F, F * expansion, bias=bias, device=device),
                nn.Linear(F * expansion, F, bias=bias, device=device)
            )
            # Initialize weights to compose to identity
            with torch.no_grad():
                # First layer: expand
                layer0 = seq[0]
                layer0.weight.zero_()
                for k in range(F * expansion):
                    i = k // expansion
                    layer0.weight[k, i] = 1.0 / (expansion ** 0.5)
                if layer0.bias is not None:
                    layer0.bias.zero_()
                
                # Second layer: compress
                layer1 = seq[1]
                layer1.weight.zero_()
                for j in range(F):
                    start = j * expansion
                    end = start + expansion
                    layer1.weight[j, start:end] = 1.0 / (expansion ** 0.5)
                if layer1.bias is not None:
                    layer1.bias.zero_()
            
            self.layers.append(seq)
        # self.norm = nn.LayerNorm(F)  # Optional

    def forward(self, x):        
        for layer in self.layers:
            x = layer(x)  # Residual connection can be added here if needed
        return x

class AsymmetricAttention(nn.Module):
    def __init__(
        self,
        dim_x: int,
        dim_y: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        update_y: bool = True,
        out_bias: bool = True,
        attention_mode: str = "flash",
        softmax_scale: Optional[float] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.attention_mode = attention_mode
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.num_heads = num_heads
        self.head_dim = dim_x // num_heads
        self.update_y = update_y
        self.softmax_scale = softmax_scale

        if dim_x % num_heads != 0:
            raise ValueError(f"dim_x={dim_x} should be divisible by num_heads={num_heads}")
        

        # Input layers.
        self.qkv_bias = qkv_bias
        self.qkv_x = nn.Linear(dim_x, 3 * dim_x, bias=qkv_bias, device=device) 
        self.qkv_y = nn.Linear(dim_y, 3 * dim_x, bias=qkv_bias, device=device)

        # Query and key normalization for stability.
        assert qk_norm
        self.q_norm_x = RMSNorm(self.head_dim, device=device)
        self.k_norm_x = RMSNorm(self.head_dim, device=device)
        self.q_norm_y = RMSNorm(self.head_dim, device=device)
        self.k_norm_y = RMSNorm(self.head_dim, device=device)

        # Output layers. y features go back down from dim_x -> dim_y.
        self.proj_x = nn.Linear(dim_x, dim_x, bias=out_bias, device=device)
        self.proj_y = nn.Linear(dim_x, dim_y, bias=out_bias, device=device) if update_y else nn.Identity()


    def run_qkv_y(self, y):
        # cp_rank, cp_size = cp.get_cp_rank_size()
        local_heads = self.num_heads 

        qkv_y = self.qkv_y(y)  # (B, L, 3 * dim)

        qkv_y = qkv_y.contiguous().view(qkv_y.size(0), qkv_y.size(1), 3, local_heads, self.head_dim)
        q_y, k_y, v_y = qkv_y.unbind(2)
        return q_y, k_y, v_y

    def prepare_qkv(
        self,
        x: torch.Tensor,  # (B, N, dim_x)
        y: torch.Tensor,  # (B, L, dim_y)
        *,
        scale_x: torch.Tensor,
        scale_y: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        valid_token_indices: torch.Tensor,
        video_shape: tuple,  # (B, T, pH, pW, D) TODO
    ):
        # # Pre-norm for visual features
        # x = modulated_rmsnorm(x, scale_x)  # (B, M, dim_x) where M = N / cp_group_size
        
        # TODO Reshape for FrameMixer using provided dimensions
        B, T, pH, pW, D = video_shape

        # TODO Pre-norm for visual features
        x = modulated_rmsnorm(x, scale_x)  # (B, M, dim_x) where M = N / cp_group_size

        # ipdb.set_trace()
        # Process visual features
        qkv_x = self.qkv_x(x)  # (B, M, 3 * dim_x)
        # ipdb.set_trace()
        assert qkv_x.dtype == torch.bfloat16
        qkv_x = all_to_all_collect_tokens(qkv_x, self.num_heads)  # (3, B, N, local_h, head_dim)

        # ipdb.set_trace()
        # Process text features
        y = modulated_rmsnorm(y, scale_y)  # (B, L, dim_y)
        q_y, k_y, v_y = self.run_qkv_y(y)  # (B, L, local_heads, head_dim)
        q_y = self.q_norm_y(q_y)
        k_y = self.k_norm_y(k_y)

        # ipdb.set_trace()
        # Split qkv_x into q, k, v
        q_x, k_x, v_x = qkv_x.unbind(0)  # (B, N, local_h, head_dim)

        q_x = self.q_norm_x(q_x) #TODO 这里面有torch.empty操作有问题会导致nan

        q_x = apply_rotary_emb_qk_real(q_x, rope_cos, rope_sin)
        k_x = self.k_norm_x(k_x)
        k_x = apply_rotary_emb_qk_real(k_x, rope_cos, rope_sin)

        # ipdb.set_trace()
        # Unite streams
        qkv = unify_streams(
            q_x,
            k_x,
            v_x,
            q_y,
            k_y,
            v_y,
            valid_token_indices,
        )
        # ipdb.set_trace()
        return qkv

    def flash_attention(self, qkv, cu_seqlens, max_seqlen_in_batch, total, local_dim):
        # ipdb.set_trace()
        with torch.autocast("cuda", enabled=False):
            out: torch.Tensor = flash_varlen_qkvpacked_attn(
                qkv,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen_in_batch,
                dropout_p=0.0,
                softmax_scale=self.softmax_scale,
            )  # (total, local_heads, head_dim)
            return out.contiguous().view(total, local_dim)

    def sdpa_attention(self, qkv):
        q, k, v = rearrange(qkv, "(b s) t h d -> t b h s d", b=1)
        with torch.autocast("cuda", enabled=False):
            with sdpa_attn_ctx():
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
                return rearrange(out, "b h s d -> s (b h d)")

    def sage_attention(self, qkv):
        q, k, v = rearrange(qkv, "(b s) t h d -> t b h s d", b=1)
        with torch.autocast("cuda", enabled=False):
            out = sage_attn(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
            return rearrange(out, "b h s d -> s (b h d)")

    def comfy_attention(self, qkv):
        q, k, v = rearrange(qkv, "(b s) t h d -> t b h s d", b=1)
        with torch.autocast("cuda", enabled=False):
            out = comfy_attn(q, k, v, heads=self.num_heads, skip_reshape=True)
            return out.squeeze(0)

    # @torch.compiler.disable()
    def run_attention(
        self,
        qkv: torch.Tensor,  # (total <= B * (N + L), 3, local_heads, head_dim)
        *,
        B: int,
        L: int,
        M: int,
        cu_seqlens: torch.Tensor,
        max_seqlen_in_batch: int,
        valid_token_indices: torch.Tensor,
    ):
        # _, cp_size = cp.get_cp_rank_size()
        N =  M
        # assert self.num_heads % cp_size == 0
        local_heads = self.num_heads
        local_dim = local_heads * self.head_dim
        total = qkv.size(0)

        if self.attention_mode != "flash":
            assert B == 1, f"Non-flash attention only supports batch size 1, got {B}"

        # ipdb.set_trace()
        if self.attention_mode == "flash":
            out = self.flash_attention(qkv, cu_seqlens, max_seqlen_in_batch, total, local_dim)
        elif self.attention_mode == "sdpa":
            out = self.sdpa_attention(qkv)
        elif self.attention_mode == "sage":
            out = self.sage_attention(qkv)
        elif self.attention_mode == "comfy":
            out = self.comfy_attention(qkv)

        # ipdb.set_trace()
        x, y = pad_and_split_xy(out, valid_token_indices, B, N, L, qkv.dtype)
        assert x.size() == (B, N, local_dim)
        assert y.size() == (B, L, local_dim)

        # ipdb.set_trace()
        x = x.contiguous().view(B, N, local_heads, self.head_dim)
        x = all_to_all_collect_heads(x)  # (B, M, dim_x = num_heads * head_dim)
        x = self.proj_x(x)  # (B, M, dim_x)
        # ipdb.set_trace()
        # if cp.is_cp_active():
        #     y = cp.all_gather(y)  # (cp_size * B, L, local_heads * head_dim)
        #     y = rearrange(y, "(G B) L D -> B L (G D)", G=cp_size, D=local_dim)  # (B, L, dim_x)
        y = self.proj_y(y)  # (B, L, dim_y)
        # ipdb.set_trace()
        return x, y

    def forward(
        self,
        x: torch.Tensor,  # (B, N, dim_x)
        y: torch.Tensor,  # (B, L, dim_y)
        *,
        scale_x: torch.Tensor,  # (B, dim_x), modulation for pre-RMSNorm.
        scale_y: torch.Tensor,  # (B, dim_y), modulation for pre-RMSNorm.
        packed_indices: Dict[str, torch.Tensor] = None,
        video_shape: tuple,  # (B, T, pH, pW, D) TODO
        **rope_rotation,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of asymmetric multi-modal attention.

        Args:
            x: (B, N, dim_x) tensor for visual tokens
            y: (B, L, dim_y) tensor of text token features
            packed_indices: Dict with keys for Flash Attention
            num_frames: Number of frames in the video. N = num_frames * num_spatial_tokens

        Returns:
            x: (B, N, dim_x) tensor of visual tokens after multi-modal attention
            y: (B, L, dim_y) tensor of text token features after multi-modal attention
        """
        B, L, _ = y.shape
        _, M, _ = x.shape

        # Predict a packed QKV tensor from visual and text features.
        # Don't checkpoint the all_to_all.
        # ipdb.set_trace()
        qkv = self.prepare_qkv(
            x=x,
            y=y,
            scale_x=scale_x,
            scale_y=scale_y,
            rope_cos=rope_rotation.get("rope_cos"),
            rope_sin=rope_rotation.get("rope_sin"),
            valid_token_indices=packed_indices["valid_token_indices_kv"],
            video_shape = video_shape # TODO
        )  # (total <= B * (N + L), 3, local_heads, head_dim)

        # ipdb.set_trace()
        x, y = self.run_attention(
            qkv,
            B=B,
            L=L,
            M=M,
            cu_seqlens=packed_indices["cu_seqlens_kv"],
            max_seqlen_in_batch=packed_indices["max_seqlen_in_batch_kv"],
            valid_token_indices=packed_indices["valid_token_indices_kv"],
        )
        return x, y


# @torch.compile(disable=not COMPILE_MMDIT_BLOCK)
class AsymmetricJointBlock(nn.Module):
    def __init__(
        self,
        hidden_size_x: int,
        hidden_size_y: int,
        num_heads: int,
        *,
        mlp_ratio_x: float = 8.0,  # Ratio of hidden size to d_model for MLP for visual tokens.
        mlp_ratio_y: float = 4.0,  # Ratio of hidden size to d_model for MLP for text tokens.
        update_y: bool = True,  # Whether to update text tokens in this block.
        device: Optional[torch.device] = None,
        **block_kwargs,
    ):
        super().__init__()
        self.update_y = update_y
        self.hidden_size_x = hidden_size_x
        self.hidden_size_y = hidden_size_y
        self.mod_x = nn.Linear(hidden_size_x, 4 * hidden_size_x, device=device)
        if self.update_y:
            self.mod_y = nn.Linear(hidden_size_x, 4 * hidden_size_y, device=device)
        else:
            self.mod_y = nn.Linear(hidden_size_x, hidden_size_y, device=device)

        # Self-attention:
        self.attn = AsymmetricAttention(
            hidden_size_x,
            hidden_size_y,
            num_heads=num_heads,
            update_y=update_y,
            device=device,
            **block_kwargs,
        )

        # MLP.
        mlp_hidden_dim_x = int(hidden_size_x * mlp_ratio_x)
        assert mlp_hidden_dim_x == int(1536 * 8)
        self.mlp_x = FeedForward(
            in_features=hidden_size_x,
            hidden_size=mlp_hidden_dim_x,
            multiple_of=256,
            ffn_dim_multiplier=None,
            device=device,
        )

        # MLP for text not needed in last block.
        if self.update_y:
            mlp_hidden_dim_y = int(hidden_size_y * mlp_ratio_y)
            self.mlp_y = FeedForward(
                in_features=hidden_size_y,
                hidden_size=mlp_hidden_dim_y,
                multiple_of=256,
                ffn_dim_multiplier=None,
                device=device,
            )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        y: torch.Tensor,
        **attn_kwargs,
    ):
        """Forward pass of a block.

        Args:
            x: (B, N, dim) tensor of visual tokens
            c: (B, dim) tensor of conditioned features
            y: (B, L, dim) tensor of text tokens
            num_frames: Number of frames in the video. N = num_frames * num_spatial_tokens

        Returns:
            x: (B, N, dim) tensor of visual tokens after block
            y: (B, L, dim) tensor of text tokens after block
        """
        N = x.size(1)


        B, T, pH, pW, D  = c.shape
        # adapt shape of c to c_x TODO
        c_x = rearrange(c, "B T pH pW D -> B (T pH pW) D")
        c_x = F.silu(c_x)
        mod_x = self.mod_x(c_x)
        scale_msa_x, gate_msa_x, scale_mlp_x, gate_mlp_x = mod_x.chunk(4, dim=2)
        """ should modify c if use c shape of (B T dim_x) or (B, T*H*W, dim_x) to average on Temporal dimension, text do not have frame dependences
         can't modify y's shape, should keep (B, L, dim_y), L is 256, because in attention 
         it's direect concatenation after map the dim_y to dim_x by a linear layer
         i.e., q = torch.cat([q_x, q_y], dim=1), k = torch.cat([k_x, k_y], dim=1), v = torch.cat([v_x, v_y], dim=1)"""
        # adapt shape of c to c_y TODO
        c_y = torch.mean(c[:,:,0,0], 1, True)
        c_y = F.silu(c_y)
        mod_y = self.mod_y(c_y)

        # c = F.silu(c)
        # mod_x = self.mod_x(c)
        # scale_msa_x, gate_msa_x, scale_mlp_x, gate_mlp_x = mod_x.chunk(4, dim=1)
        # mod_y = self.mod_y(c)
        #  if self.update_y:
        #     scale_msa_y, gate_msa_y, scale_mlp_y, gate_mlp_y = mod_y.chunk(4, dim=1)
        # else:
        #     scale_msa_y = mod_y

        # ipdb.set_trace()
        if self.update_y:
            scale_msa_y, gate_msa_y, scale_mlp_y, gate_mlp_y = mod_y.chunk(4, dim=2) # TODO
        else:
            scale_msa_y = mod_y


        # ipdb.set_trace()
        # Self-attention block.
        x_attn, y_attn = self.attn(
            x,
            y,
            scale_x=scale_msa_x,
            scale_y=scale_msa_y,
            video_shape = (B,T,pH,pW,D), # TODO
            **attn_kwargs,
        )
        # ipdb.set_trace()
        assert x_attn.size(1) == N
        x = residual_tanh_gated_rmsnorm(x, x_attn, gate_msa_x)
        if self.update_y:
            y = residual_tanh_gated_rmsnorm(y, y_attn, gate_msa_y)
        # ipdb.set_trace()
        # MLP block.
        x = self.ff_block_x(x, scale_mlp_x, gate_mlp_x)
        if self.update_y:
            y = self.ff_block_y(y, scale_mlp_y, gate_mlp_y)
        # ipdb.set_trace()

        return x, y

    def ff_block_x(self, x, scale_x, gate_x):
        x_mod = modulated_rmsnorm(x, scale_x)
        x_res = self.mlp_x(x_mod)
        x = residual_tanh_gated_rmsnorm(x, x_res, gate_x)  # Sandwich norm
        return x

    def ff_block_y(self, y, scale_y, gate_y):
        y_mod = modulated_rmsnorm(y, scale_y)
        y_res = self.mlp_y(y_mod)
        y = residual_tanh_gated_rmsnorm(y, y_res, gate_y)  # Sandwich norm
        return y


# @torch.compile(disable=not COMPILE_FINAL_LAYER)
class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(
        self,
        hidden_size,
        patch_size,
        out_channels,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, device=device)
        self.mod = nn.Linear(hidden_size, 2 * hidden_size, device=device)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, device=device)

    def forward(self, x, c):
        # c = F.silu(c)
        # shift, scale = self.mod(c).chunk(2, dim=1)
        B, T, pH, pW, D  = c.shape #TODO
        # adapt shape of c to c_x TODO
        c_x = rearrange(c, "B T pH pW D -> B (T pH pW) D")
        c_x = F.silu(c_x)
        shift, scale = self.mod(c_x).chunk(2, dim=2)

        # shift, scale = self.mod(c).chunk(2, dim=1)

        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class AsymmDiTJoint(nn.Module):
    """
    Diffusion model with a Transformer backbone.

    Ingests text embeddings instead of a label.
    """

    def __init__(
        self,
        *,
        patch_size=2,
        in_channels=4,
        hidden_size_x=1152,
        hidden_size_y=1152,
        depth=48,
        num_heads=16,
        mlp_ratio_x=8.0,
        mlp_ratio_y=4.0,
        t5_feat_dim: int = 4096,
        t5_token_length: int = 256,
        patch_embed_bias: bool = True,
        timestep_mlp_bias: bool = True,
        timestep_scale: Optional[float] = None,
        use_extended_posenc: bool = False,
        rope_theta: float = 10000.0,
        device: Optional[torch.device] = None,
        **block_kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size_x = hidden_size_x
        self.hidden_size_y = hidden_size_y
        self.head_dim = hidden_size_x // num_heads  # Head dimension and count is determined by visual.
        self.use_extended_posenc = use_extended_posenc
        self.t5_token_length = t5_token_length
        self.t5_feat_dim = t5_feat_dim
        self.rope_theta = rope_theta  # Scaling factor for frequency computation for temporal RoPE.

        self.x_embedder = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=hidden_size_x,
            bias=patch_embed_bias,
            device=device,
        )
        # Conditionings
        # Timestep
        self.t_embedder = TimestepEmbedder(hidden_size_x, bias=timestep_mlp_bias, timestep_scale=timestep_scale)

        # Caption Pooling (T5)
        self.t5_y_embedder = AttentionPool(t5_feat_dim, num_heads=8, output_dim=hidden_size_x, device=device)

        # Dense Embedding Projection (T5)
        self.t5_yproj = nn.Linear(t5_feat_dim, hidden_size_y, bias=True, device=device)

        # Initialize pos_frequencies as an empty parameter.
        # self.pos_frequencies = nn.Parameter(torch.empty(3, self.num_heads, self.head_dim // 2, device=device))
        self.pos_frequencies = nn.Parameter(torch.ones(3, self.num_heads, self.head_dim // 2, device=device)*0.5)

        # for depth 48:
        #  b =  0: AsymmetricJointBlock, update_y=True
        #  b =  1: AsymmetricJointBlock, update_y=True
        #  ...
        #  b = 46: AsymmetricJointBlock, update_y=True
        #  b = 47: AsymmetricJointBlock, update_y=False. No need to update text features.
        blocks = []
        for b in range(depth):
            # Joint multi-modal block
            update_y = b < depth - 1
            block = AsymmetricJointBlock(
                hidden_size_x,
                hidden_size_y,
                num_heads,
                mlp_ratio_x=mlp_ratio_x,
                mlp_ratio_y=mlp_ratio_y,
                update_y=update_y,
                device=device,
                **block_kwargs,
            )

            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

        self.final_layer = FinalLayer(hidden_size_x, patch_size, self.out_channels, device=device)

    def embed_x(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C=12, T, H, W) tensor of visual tokens

        Returns:
            x: (B, C=3072, N) tensor of visual tokens with positional embedding.
        """
        return self.x_embedder(x)  # Convert BcTHW to BCN

    # @torch.compile(disable=not COMPILE_MMDIT_BLOCK)
    def prepare(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        t5_feat: torch.Tensor,
        t5_mask: torch.Tensor,
    ):
        """Prepare input and conditioning embeddings."""

        with torch.profiler.record_function("x_emb_pe"):
            # Visual patch embeddings with positional encoding.
            T, H, W = x.shape[-3:]
            pH, pW = H // self.patch_size, W // self.patch_size
            x = self.embed_x(x)  # (B, N, D), where N = T * H * W / patch_size ** 2
            assert x.ndim == 3
            B = x.size(0)
        # ipdb.set_trace()
        with torch.profiler.record_function("rope_cis"):
            # Construct position array of size [N, 3].
            # pos[:, 0] is the frame index for each location,
            # pos[:, 1] is the row index for each location, and
            # pos[:, 2] is the column index for each location.
            pH, pW = H // self.patch_size, W // self.patch_size
            N = T * pH * pW
            assert x.size(1) == N
            pos = create_position_matrix(T, pH=pH, pW=pW, device=x.device, dtype=torch.float32)  # (N, 3)
            rope_cos, rope_sin = compute_mixed_rotation(
                freqs=self.pos_frequencies, pos=pos
            )  # Each are (N, num_heads, dim // 2)

        # ipdb.set_trace()
        with torch.profiler.record_function("t_emb"):
            # Global vector embedding for conditionings.
            c_t = self.t_embedder(1 - sigma)  # (B, D)

        with torch.profiler.record_function("t5_pool"):
            # Pool T5 tokens using attention pooler
            # Note y_feat[1] contains T5 token features.
            assert (
                t5_feat.size(1) == self.t5_token_length
            ), f"Expected L={self.t5_token_length}, got {t5_feat.shape} for y_feat."
            t5_y_pool = self.t5_y_embedder(t5_feat, t5_mask)  # (B, D)
            assert t5_y_pool.size(0) == B, f"Expected B={B}, got {t5_y_pool.shape} for t5_y_pool."
       

        t5_y_pool = t5_y_pool.unsqueeze(1).expand(B, c_t.shape[1], t5_y_pool.shape[1])
        c = c_t + t5_y_pool
        
        # c = c.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, T, pH, pW, 1) # TODO
        c = c.unsqueeze(2).unsqueeze(3).repeat(1, 1, pH, pW, 1) # TODO
        # c = rearrange(c, "B T pH pW D -> B (T pH pW) D")
        # c = c.unsqueeze(1).repeat(1, T, 1)

        y_feat = self.t5_yproj(t5_feat)  # (B, L, t5_feat_dim) --> (B, L, D)

        return x, c, y_feat, rope_cos, rope_sin

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        y_feat: List[torch.Tensor],
        y_mask: List[torch.Tensor],
        packed_indices: Dict[str, torch.Tensor] = None,
        rope_cos: torch.Tensor = None,
        rope_sin: torch.Tensor = None,
    ):
        """Forward pass of DiT.

        Args:
            x: (B, C, T, H, W) tensor of spatial inputs (images or latent representations of images)
            sigma: (B,) tensor of noise standard deviations
            y_feat: List((B, L, y_feat_dim) tensor of caption token features. For SDXL text encoders: L=77, y_feat_dim=2048)
            y_mask: List((B, L) boolean tensor indicating which tokens are not padding)
            packed_indices: Dict with keys for Flash Attention. Result of compute_packed_indices.
        """
        B, _, T, H, W = x.shape
        with sdpa_kernel(torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION):
            x, c, y_feat, rope_cos, rope_sin = self.prepare(x, sigma, y_feat[0], y_mask[0])
        del y_mask

        N = x.size(1)
        M = N 

        for i, block in enumerate(self.blocks):
            # print(f"\nBlock {i}:")
            x_prev, y_prev = x, y_feat  # Store previous values for debugging
            
            x, y_feat = block(
                x,
                c,
                y_feat,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                packed_indices=packed_indices,
            )
            
        del y_feat  # Final layers don't use dense text features.
        
        # Final layer processing
        x = self.final_layer(x, c)

        patch = x.size(2)

        # First make the input tensor contiguous
        x = x.contiguous()

        # Perform rearrange and immediately create a new tensor
        x = rearrange(
            x,
            "B (T hp wp) (p1 p2 c) -> B c T (hp p1) (wp p2)",
            T=T,
            hp=H // self.patch_size,
            wp=W // self.patch_size,
            p1=self.patch_size,
            p2=self.patch_size,
            c=self.out_channels,
        ).contiguous()  # Force new memory allocation

        return x