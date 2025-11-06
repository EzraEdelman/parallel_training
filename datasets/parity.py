# rn feel like making this a torch.utils.data.Dataset is unneeded and might add overhead (and I'm lazy rn)
import jax
from functools import partial
import jax.numpy as jnp
import jax.random as jr
from typing import Tuple, NamedTuple


class ParityConfig(NamedTuple):
    d: int = 10
    k: int = 2
    dtype: jnp.dtype = jnp.int32
    zero_one: bool = False

class Parity:
    def __init__(self, d=10, k=2, dtype=jnp.int32, zero_one=False):
        self.d = d
        self.k = k
        self.dtype = dtype
        self.zero_one = zero_one 

        if not self.zero_one:
            self.real_create_batch = lambda *args: self._create_batch(*args) * 2 - 1
            self.evaluate_parity = lambda x: jnp.prod(x, axis=-1)
        else:
            self.real_create_batch = lambda *args: self._create_batch(*args)
            self.evaluate_parity = lambda x: jnp.sum(x[..., :self.k], axis=-1) % 2

    
    def _create_batch(self, key, n: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Returns:
      X_train : (n, d)
      y_train : (n,)
    """
        X = jax.random.bernoulli(key, p=0.5, shape=(n, self.d)).astype(self.dtype)
        y = self.evaluate_parity(X).astype(self.dtype)
        return X, y
    
    def create_batch(self, key, n: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self.real_create_batch(key, n)
    


    # @partial(jax.jit, static_argnums=(2,3))
    def create_batches(self, key, n: int, num_seeds: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Create batches for multiple seeds"""
        keys = jr.split(key, num_seeds)
        return jax.vmap(lambda key_i: self.create_batch(key_i, n))(keys)
    
    @staticmethod
    def config(**kwargs):
        return ParityConfig(**kwargs)

