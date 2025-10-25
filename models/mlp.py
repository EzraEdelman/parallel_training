import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.lax as lax
import dataclasses
from jax.nn import initializers
from typing import Callable, List
from functools import partial
def linear(in_features, out_features, *args,key, kernel_init=initializers.lecun_normal(), bias_init=initializers.zeros, **kwargs):

    key1, key2, key3 = jr.split(key, 3)
    out = eqx.nn.Linear(in_features, out_features, *args, key=key1, **kwargs)
    # out = eqx.tree_at(lambda l: l.bias, out, replace_fn=lambda b: bias_init(key2, b.shape))
    # out = eqx.tree_at(lambda l: l.weight, out, replace_fn=lambda w: kernel_init(key3, w.shape))
    return out



class MLP(eqx.Module):
  layers: List[eqx.nn.Linear]
  def __init__(self, config, key):
    _linear = partial(linear, kernel_init=config.kernel_init, bias_init=config.bias_init) 
    if config.hidden_layers == 0:
      self.layers = [_linear(config.in_dim, config.out_dim, key)]
    else:
      keys = jr.split(key, config.hidden_layers + 1)
      self.layers = [_linear(config.in_dim, config.hidden_dim, key=keys[0])]
      for i in range(config.hidden_layers - 1):
        self.layers.append(_linear(config.hidden_dim, config.hidden_dim, key=keys[i + 1]))
      self.layers.append(_linear(config.hidden_dim, config.out_dim, key=keys[-1]))
    
  @staticmethod
  def config(**kwargs):
    return MLPConfig(**kwargs)
  
  def __call__(self, x, key):
    for layer in self.layers:
      x = layer(x)
    return x
  

@dataclasses.dataclass(unsafe_hash=True)
class MLPConfig:
  in_dim: int
  hidden_dim: int = 16
  out_dim: int = 1
  hidden_layers: int = 2
  kernel_init: Callable = initializers.lecun_normal()
  bias_init: Callable = initializers.zeros
  dtype: jnp.dtype = jnp.bfloat16

  def replace(self, **kwargs):
    return dataclasses.replace(self, **kwargs)