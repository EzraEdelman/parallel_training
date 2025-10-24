# rn feel like making this a torch.utils.data.Dataset is unneeded and might add overhead (and I'm lazy rn)
import jax
from functools import partial
import jax.numpy as jnp
import jax.random as jr
from typing import Tuple
import dataclasses


@dataclasses.dataclass(unsafe_hash=True)
class ParityConfig:
    d: int = 10
    k: int = 2
    dtype: jnp.dtype = jnp.int32

    def replace(self, **kwargs):
        return dataclasses.replace(self, **kwargs)


class Parity:
    def __init__(self, config: ParityConfig):
        self.d = config.d
        self.k = config.k
        self.dtype = config.dtype

    def create_batch(self, key, n: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Returns:
      X_train : (n, d)
      y_train : (n,)
    """
        X = jax.random.bernoulli(key, p=0.5, shape=(n, self.d)).astype(self.dtype)
        y = self.evaluate_parity(X).astype(self.dtype)
        return X, y

    def evaluate_parity(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(x[..., :self.k], axis=-1) % 2

    # @partial(jax.jit, static_argnums=(2,3))
    def create_batches(self, key, n: int, num_seeds: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Create batches for multiple seeds"""
        keys = jr.split(key, num_seeds)
        return jax.vmap(lambda key_i: self.create_batch(key_i, n))(keys)
    
    @staticmethod
    def config(**kwargs):
        return ParityConfig(**kwargs)

    

