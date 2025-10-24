import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.lax as lax
import dataclasses

class Linear(eqx.Module):
  def __init__(self, in_dim, out_dim, key, kernel_init=eqx.nn.initializers.lecun_normal(), bias_init=eqx.nn.initializers.zeros_init(), dtype=jnp.bfloat16):
    self.weight = kernel_init(in_dim, out_dim, key=key, dtype=dtype)
    self.bias = bias_init(out_dim, key=key, dtype=dtype)
  
  def __call__(self, x):
    return x @ self.weight + self.bias

class MLP(eqx.Module):
  def __init__(self, config, key):
    args = {"dtype": config.dtype, "kernel_init": config.kernel_init, "bias_init": config.bias_init}
    if config.hidden_layers == 0:
      self.layers = [Linear(config.in_dim, config.out_dim, key, **args)]
    else:
      keys = jr.split(key, config.hidden_layers + 1)
      self.layers = [Linear(config.in_dim, config.hidden_dim, key=keys[0], **args)]
      for i in range(config.hidden_layers - 1):
        self.layers.append(Linear(config.hidden_dim, config.hidden_dim, key=keys[i + 1], **args))
      self.layers.append(Linear(config.hidden_dim, config.out_dim, key=keys[-1], **args))

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
  kernel_init: eqx.nn.Initializer = eqx.nn.initializers.lecun_normal()
  bias_init: eqx.nn.Initializer = eqx.nn.initializers.zeros_init()
  dtype: jnp.dtype = jnp.bfloat16

  def replace(self, **kwargs):
    return dataclasses.replace(self, **kwargs)