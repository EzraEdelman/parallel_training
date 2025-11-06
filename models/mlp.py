import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.lax as lax
import dataclasses
from jax.nn import initializers
from typing import Callable, List, NamedTuple, Dict
from functools import partial
import optax

def linear(in_features, out_features, *args,key, kernel_init=initializers.lecun_normal(), bias_init=initializers.zeros, **kwargs):

    key1, key2, key3 = jr.split(key, 3)
    out = eqx.nn.Linear(in_features, out_features, *args, key=key1, **kwargs)
    # out = eqx.tree_at(lambda l: l.bias, out, replace_fn=lambda b: bias_init(key2, b.shape))
    # out = eqx.tree_at(lambda l: l.weight, out, replace_fn=lambda w: kernel_init(key3, w.shape))
    return out

class MLP(eqx.Module):
  layers: List[eqx.nn.Linear]

  # @staticmethod
  # def create(config, key):
  #   config = {k:jax.tree_util.Partial(v) if callable(v) else v for k,v in config.items()}
  #   axis = {k:0 if k=="layer_init_scale" else None for k in config.keys()}
  #   return jax.vmap(MLP, in_axes=(axis, None))( config, key)
  
  def __init__(self, key, in_dim, out_dim, hidden_layers, hidden_dim, bias_init=jax.nn.initializers.zeros, kernel_init=jax.nn.initializers.lecun_normal(), layer_init_scale=[1,1,1], dtype=jnp.float32):
    _linear = partial(linear, bias_init=bias_init) 
    if hidden_layers == 0:
      self.layers = [_linear(in_dim, out_dim, key, kernel_init=lambda *args: layer_init_scale[0]*kernel_init(*args))]
    else:
      keys = jr.split(key, hidden_layers + 1)
      self.layers = [_linear(in_dim, hidden_dim, key=keys[0], kernel_init=lambda *args: layer_init_scale[0]*kernel_init(*args))]
      for i in range(hidden_layers - 1):
        self.layers.append(_linear(hidden_dim, hidden_dim, key=keys[i + 1], kernel_init=lambda *args: layer_init_scale[1]*kernel_init(*args)))
      self.layers.append(_linear(hidden_dim, out_dim, key=keys[-1], kernel_init=lambda *args: layer_init_scale[-1]*kernel_init(*args) ))

  
  def __call__(self, x, key):
    for layer in self.layers:
      x = layer(x)
    return x
  
  @staticmethod
  def config(**kwargs):
    return MLPConfig(**kwargs)
  
  @staticmethod
  def optimizer_config(**kwargs):
    return optimizer_config(**kwargs)
  # @staticmethod
  # def config(**kwargs):
  #   return {
  #   "in_dim": kwargs.get("in_dim", 10),
  #   "hidden_dim": kwargs.get("hidden_dim", 16),
  #   "out_dim": kwargs.get("out_dim", 1),
  #   "hidden_layers": kwargs.get("hidden_layers", 2),
  #   "kernel_init": kwargs.get("kernel_init", initializers.lecun_normal()),
  #   "bias_init": kwargs.get("bias_init", initializers.zeros),
  #   "dtype": kwargs.get("dtype", jnp.bfloat16),
  #   "layer_init_scale": kwargs.get("layer_init_scale", [1.0]*3),
  #   "criterion": kwargs.get("criterion", lambda y_pred, y: optax.squared_error(y_pred, y).mean()),
  #   "optimizer": kwargs.get("optimizer", optax.sgd),
  #   "optimizer_kwargs": kwargs.get("optimizer_kwargs", {"learning_rate": 1e-2}),
  #   "model": MLP,
  # }
class optimizer_config(NamedTuple):
    learning_rate: jax.typing.ArrayLike = jnp.array(1e-2)
    optimizer: Callable = optax.adamw

class MLPConfig(NamedTuple):
    key: jax.Array = jr.PRNGKey(0)
    in_dim: int = 10
    out_dim: int = 1
    hidden_dim: int = 16
    hidden_layers: int = 2
    kernel_init: Callable = initializers.lecun_normal()
    bias_init: Callable = initializers.zeros
    layer_init_scale: List[float] = [1.0]*3
    dtype: jnp.dtype = jnp.float32
    # model: eqx.Module = MLP
    # criterion: Callable = lambda y_pred, y: optax.squared_error(y_pred, y).mean()

    # def __getitem__(self, key):
    #     return self.key