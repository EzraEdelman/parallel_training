import matplotlib.pyplot as plt
import equinox as eqx
import optax
import jax.numpy as jnp
import jax
import jax.random as jr
from functools import partial
from tqdm.auto import tqdm
from typing import Tuple, List, Dict, Any, Callable, NamedTuple
import dataclasses
from models.transformer import Transformer
from datasets.parity import Parity
from jax_tqdm import scan_tqdm
from models.mlp import MLP
def _check_axis(leaf, ref_leaf):
    if isinstance(leaf, jax.typing.ArrayLike):
        diff = len(jnp.shape(leaf)) - len(jnp.shape(ref_leaf))
        if diff > 0:
            return diff
    return None
def wrap(obj, args, default_args):
    num_vmaps = jax.tree.map(_check_axis, args, default_args)
    axes = []

    for path, num in jax.tree.leaves_with_path(num_vmaps):
        axis = jax.tree.map_with_path(lambda _path, leaf: 0 if _path == path else None,num_vmaps)
        axes.extend([axis]*num)
    
    create = lambda kwargs: obj(*kwargs)
    for axis in axes:
        create = jax.vmap(create, in_axes=[axis])
    # print(args)
    obj = create(args)
    axes = axes
    def apply(f, parallel_kwargs, kwargs):
        axis = {}
        for key in parallel_kwargs:
            axis[key] = 0
            for key in kwargs:
                axis[key] = None
        
        def wrapped_f(parallel_args, args):
            print("compiling")
            return f(**parallel_args, **args)

        for axis in axes:
            wrapped_f = jax.vmap(wrapped_f, in_axes=[0, None])
        return wrapped_f(parallel_kwargs, kwargs)
    return obj, apply


def test(model, data, keys):
    return jax.vmap(model, in_axes=(0, None))(data, keys)
dataset_config = Parity.config(d=20, k=6, zero_one=False)
model, apply = wrap(MLP, MLP.config(in_dim=20, layer_init_scale=[jnp.arange(5), 1.,1.]),  MLP.config())
dataset = Parity(dataset_config)
keys = jr.split(jr.PRNGKey(0), 5)
data = dataset.create_batch(jr.PRNGKey(0), 1)[0]
print(apply(test, parallel_kwargs={'model':model}, kwargs={'data':data, 'keys':keys}))
print(apply(test, parallel_kwargs={'model':model}, kwargs={'data':data, 'keys':keys}))