import matplotlib.pyplot as plt
import equinox as eqx
import optax
import jax.numpy as jnp
import jax
import jax.random as jr
from functools import partial
from tqdm.auto import tqdm
from typing import Any, Callable
import dataclasses
from models.transformer import Transformer
from datasets.parity import Parity
from jax_tqdm import scan_tqdm


@dataclasses.dataclass(unsafe_hash=True)
class TrainConfig:
    lrs: jax.Array
    num_seeds: int
    criterion: Callable
    model: type
    dataset: type
    model_config: Any = None
    num_steps: int = 5000
    rng_key: jax.Array = dataclasses.field(default_factory=lambda: jr.PRNGKey(0))
    batch_size: int = 32
    trainset_size: int = -1 # -1 for online dataset
    dataset_config: Any = None

    def replace(self, **kwargs):
        return dataclasses.replace(self, **kwargs)


def compute_metrics(logits, labels, loss):
    """Compute accuracy and loss from predictions"""
    preds = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(preds == labels)
    return {'accuracy': accuracy, 'loss': loss}


def init_models(config: TrainConfig, key):
    """Initialize models and optimizers for all seeds and learning rates"""
    if config.model_config is None:
        config.model_config = config.model.config()
    
    # Split keys for each seed
    keys = jr.split(key, config.num_seeds)
    
    def init_single_seed(seed_key):
        """Initialize models for all learning rates for one seed"""
        lr_keys = jr.split(seed_key, len(config.lrs))
        
        def init_single_model(model_key):
            model = config.model(config.model_config, model_key)

            optimizer = optax.adamw(1.0)  # LR will be set per-model
            opt_state = optimizer.init(model)
            # model = eqx.partition(model, eqx.is_array)
            return model, opt_state
        
        # Vmap over learning rates
        models, opt_states = jax.vmap(init_single_model)(lr_keys)
        return models, opt_states
    
    # Vmap over seeds
    models, opt_states = jax.vmap(init_single_seed)(keys)
    return models, opt_states


@partial(jax.jit, static_argnames=('criterion','num_seeds'))
def train_step(models, opt_states, data, lrs, criterion, key, num_seeds):
    """Train step vmapped over seeds and learning rates - returns models, opt_states, and metrics"""
    # Split keys for each seed
    keys = jr.split(key, num_seeds)

    @partial(jax.vmap, in_axes=(0,0,0))
    def train_seed(model, key, data):
        X, y = data
        def loss_fn(model):
            y_pred = eqx.filter_vmap(model, in_axes=(0,None))(X, key)
            return criterion(y_pred, y), y_pred
        # Further split keys for each LR
        (loss, logits), grads = jax.vmap(eqx.filter_value_and_grad(loss_fn, has_aux=True))(model)
        # Compute metrics
        metrics = jax.vmap(compute_metrics, in_axes=(0,None,0))(logits, y, loss)
        return grads, metrics
    
    grads, metrics = train_seed(models, keys, data)

    # Update model
    @partial(jax.vmap, in_axes=(0,1,1,1), out_axes=1)
    def update(lr, grads, opt_state, model):
        updates, opt_state = jax.vmap(optax.adamw(lr).update)(
            grads, opt_state, model
        )
        model = jax.vmap(eqx.apply_updates)(model, updates)
        return model, opt_state
    
    models, opt_states = update(lrs, grads, opt_states, models)
    return models, opt_states, metrics


@partial(jax.jit, static_argnames=('criterion',))
@partial(jax.vmap, in_axes=(0,None, None))
@partial(jax.vmap, in_axes=(0,None, None))
def eval_step(model, data, criterion):
    """Eval step for a single model - returns metrics"""
    X, y = data
    logits = jax.vmap(model, in_axes=(0,None))(X, None)
    loss = criterion(logits, y)
    metrics = compute_metrics(logits, y, loss)
    return metrics