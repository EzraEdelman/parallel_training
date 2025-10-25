import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.lax as lax
import dataclasses
from typing import Optional

        

class DecoderBlock(eqx.Module):
    attention: eqx.nn.MultiheadAttention
    mlp_proj: eqx.nn.Linear
    mlp_w: eqx.nn.Linear
    output_proj: eqx.nn.Linear
    # mlp_dropout: eqx.nn.Dropout
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm

    def __init__(self, config, key):
        keys = jr.split(key, 6)
        embd_dim = config.embd_dim
        num_heads = config.num_heads
        qkv_dim = config.qkv_dim
        mlp_dim = config.mlp_dim
        dtype = config.dtype
        # dropout_rate = config.dropout_rate
        
        self.attention = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=embd_dim,
            key=keys[0],
            use_query_bias=True,
            use_key_bias=True,
            use_value_bias=True,
            use_output_bias=True,
            inference=True
        )

        # self.mlp_dropout = eqx.nn.Dropout(0.5)
        self.mlp_proj = eqx.nn.Linear(embd_dim, mlp_dim, key=keys[1], dtype=dtype)
        self.mlp_w = eqx.nn.Linear(mlp_dim, embd_dim, key=keys[2], dtype=dtype)
        self.output_proj = eqx.nn.Linear(embd_dim, embd_dim, key=keys[3], dtype=dtype)
        
        self.norm1 = eqx.nn.LayerNorm(embd_dim, dtype=dtype)
        self.norm2 = eqx.nn.LayerNorm(embd_dim, dtype=dtype)


    def __call__(self, x):
        # # Create causal mask
        seq_len = x.shape[0]
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        
        # Self-attention with residual
        normed = jax.vmap(self.norm1)(x)
        attn_out = self.attention(normed, normed, normed, mask=mask, inference=True)
        x = x + attn_out
        
        # MLP with residual
        normed = jax.vmap(self.norm2)(x)
        mlp_out = jax.vmap(self.mlp_proj)(normed)
        mlp_out = jax.nn.gelu(mlp_out)
        mlp_out = jax.vmap(self.mlp_w)(mlp_out)
        

        
        x = x + mlp_out
        x = jax.vmap(self.output_proj)(x)
        
        return x

"""
outputs logits for last token only!
"""
class Transformer(eqx.Module):
    embeddings: eqx.nn.Embedding
    pos_embeddings: eqx.nn.Embedding
    layers: list
    layer_norm: eqx.nn.LayerNorm
    lm_head: eqx.nn.Linear
    num_layers: int
    def __init__(self, config, key):
        keys = jr.split(key, config.num_layers + 2)
        
        vocab_size = config.vocab_size
        embd_dim = config.embd_dim
        num_layers = config.num_layers
        max_len = config.max_len
        dtype = config.dtype
        self.num_layers = num_layers
        self.embeddings = eqx.nn.Embedding(vocab_size, embd_dim, key=keys[0], dtype=dtype)
        self.pos_embeddings = eqx.nn.Embedding(max_len, embd_dim, key=keys[1], dtype=dtype)
        
        # # Create decoder blocks
        self.layers = [
            DecoderBlock(config, keys[i + 2]) 
            for i in range(num_layers)
        ]
        # make_layer = lambda k: DecoderBlock(config, k)
        # self.layers = jax.vmap(make_layer)(keys[2:])
        # # self.layers = eqx.combine(*self.layers_split)
        
        self.layer_norm = eqx.nn.LayerNorm(embd_dim, dtype=dtype)
        self.lm_head = eqx.nn.Linear(embd_dim, vocab_size, key=keys[-1], dtype=dtype)
    
    @staticmethod
    def config(**kwargs):
        return TransformerConfig(**kwargs)

    def __call__(self, x, key):
        seq_len = x.shape[0]
        # Token embeddings + positional embeddings
        x = jax.vmap(self.embeddings)(x)
        positions = jnp.arange(seq_len) # Shape: (1, seq_len)
        pos_emb = jax.vmap(self.pos_embeddings)(positions)
        x = x + pos_emb
        
        for layer in self.layers:
            x = layer(x)
        
        x = jax.vmap(self.layer_norm)(x)
        x = jax.vmap(self.lm_head)(x)
        
        return x[-1]


@dataclasses.dataclass
class TransformerConfig:
    vocab_size: int
    embd_dim: int = 256
    num_heads: int = 2
    num_layers: int = 2
    qkv_dim: int = 256
    mlp_dim: int = 1024
    max_len: int = 10
    dtype: jnp.dtype = jnp.float32
    # dropout_rate: float = 0.0

    def replace(self, **kwargs):
        return dataclasses.replace(self, **kwargs)


if __name__ == "__main__":
    config = TransformerConfig(vocab_size=100)
    key = jr.PRNGKey(0)
    transformer = Transformer(config, key)
    
    # Test forward pass
    test_input = jnp.ones((2, 8), dtype=jnp.int32)
    output = jax.vmap(transformer, in_axes=(0, None))(test_input, jr.PRNGKey(1))
    print(f"Output shape: {output.shape}")
    print(f"Output:\n{output}")