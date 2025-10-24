from flax import nnx
import optax
import dataclasses
from typing import Any
import jax.numpy as jnp
import jax

class DecoderBlock(nnx.Module):
    def __init__(self, config, rngs):
      num_heads = config.num_heads
      embd_dim = config.embd_dim
      qkv_dim = config.qkv_dim
      mlp_dim = config.mlp_dim
      dropout_rate = config.dropout_rate
      kernel_init = config.kernel_init
      bias_init = config.bias_init
      
      self.attention = nnx.MultiHeadAttention(
          num_heads, embd_dim, qkv_dim, embd_dim, 
          decode=False, 
          kernel_init=kernel_init,
          bias_init=bias_init,
          rngs=rngs
      )
      self.mlp_proj = nnx.Linear(
          embd_dim, mlp_dim, 
          kernel_init=kernel_init,
          bias_init=bias_init,
          rngs=rngs
      )
      self.mlp_w = nnx.Linear(
          mlp_dim, embd_dim,
          kernel_init=kernel_init,
          bias_init=bias_init,
          rngs=rngs
      )
      self.output_proj = nnx.Linear(
          embd_dim, embd_dim,
          kernel_init=kernel_init,
          bias_init=bias_init,
          rngs=rngs
      )

      self.mlp_dropout = nnx.Dropout(dropout_rate, rngs=rngs)
      self.norm1 = nnx.LayerNorm(embd_dim, rngs=rngs)
      self.norm2 = nnx.LayerNorm(embd_dim, rngs=rngs)


    def __call__(self, x):
      mask = nnx.make_causal_mask(x[:, :, 0])
      x += self.attention(self.norm1(x), mask=mask)
      x += self.mlp_dropout(self.mlp_w(nnx.gelu(self.mlp_proj(self.norm2(x)))))
      
      return self.output_proj(x)
  
class Transformer(nnx.Module):
  
  def __init__(self, config, rngs):
    vocab_size = config.vocab_size
    embd_dim = config.embd_dim
    num_layers = config.num_layers
    max_len = config.max_len
    kernel_init = config.kernel_init
    bias_init = config.bias_init
    posemb_init = config.posemb_init if config.posemb_init is not None else kernel_init
    
    self.num_layers = num_layers
    self.embeddings = nnx.Embed(
        vocab_size, embd_dim,
        embedding_init=kernel_init,
        rngs=rngs
    )
    self.pos_embeddings = nnx.Embed(
        max_len, embd_dim,
        embedding_init=posemb_init,
        rngs=rngs
    )
    
    @nnx.split_rngs(splits=num_layers)
    @nnx.vmap(axis_size=num_layers)
    def create_block(rngs: nnx.Rngs):
      return DecoderBlock(config, rngs)
    self.layers = create_block(rngs)

    self.layer_norm = nnx.LayerNorm(embd_dim, rngs=rngs)
    self.lm_head = nnx.Linear(
        embd_dim, vocab_size,
        kernel_init=kernel_init,
        bias_init=bias_init,
        rngs=rngs
    )
  
  @staticmethod
  def config(**kwargs):
    return TransformerConfig(**kwargs)

  def __call__(self, x):
    batch_size, seq_len = x.shape
    
    # Token embeddings + positional embeddings
    x = self.embeddings(x)
    positions = jnp.arange(seq_len)[None, :]  # Shape: (1, seq_len)
    x = x + self.pos_embeddings(positions)
    
    @nnx.scan(in_axes=(nnx.Carry,0), out_axes=nnx.Carry)
    def forward(x, model):
      x = model(x)
      return x
    
    x = forward(x, self.layers)
    x = self.layer_norm(x)
    x = self.lm_head(x)
    return x

@dataclasses.dataclass(unsafe_hash=True)
class TransformerConfig:

  vocab_size: int
  # dtype: Any = jnp.float32
  embd_dim: int = 16
  num_heads: int = 2
  num_layers: int = 2
  qkv_dim: int = 12
  mlp_dim: int = 16
  max_len: int = 10
  dropout_rate: float = 0
  kernel_init: nnx.Initializer = nnx.initializers.lecun_normal()
  bias_init: nnx.Initializer = nnx.initializers.zeros_init()
  posemb_init: nnx.Initializer | None = None

  def replace(self, **kwargs):
    return dataclasses.replace(self, **kwargs)


# if __name__ == "__main__":
#   config = TransformerConfig(vocab_size=100)
#   rngs = nnx.Rngs(0)
#   transformer = Transformer(config, rngs)
#   print(transformer(jnp.ones((1,8), dtype=int)))