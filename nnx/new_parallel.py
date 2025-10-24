# %env JAX_CHECK_TRACER_LEAKS = 1
from models.transformer import Transformer, TransformerConfig
from flax import nnx
import optax
import jax.numpy as jnp
import jax
from functools import partial
from tqdm.auto import tqdm
from typing import Tuple, List, Dict, Any, Callable
import dataclasses


@dataclasses.dataclass(unsafe_hash=True)
class TrainConfig:
  lrs: jax.Array
  num_seeds: int
  criterion: Callable
  model: nnx.Module
  model_config: Any = False

  def replace(self, **kwargs):
    return dataclasses.replace(self, **kwargs)


def init_model(config: TrainConfig):
  rngs = nnx.Rngs(params=0, dropout=1)
  if not config.model_config:
    config.model_config = config.model.config()
  @nnx.jit
  @nnx.split_rngs(splits=(config.num_seeds,))
  @nnx.vmap(in_axes=(nnx.StateAxes({(nnx.Param,'params', 'dropout'): 0, ...: None}),None))
  @nnx.vmap(in_axes=(None,0,))
  def _init_model(rngs, lr):
      model = config.model(config.model_config, rngs)
      optimizer = nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)

      metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(threshold=None),
        loss=nnx.metrics.Average('loss'),
      )
      return model, nnx.state(optimizer), metrics
  return _init_model(rngs, config.lrs)



@partial(jax.jit, static_argnames=('criterion'))
@nnx.vmap(in_axes=(0,0,0, 0, 0, None, None)) # data seeds #in_axes=(state_axes, nnx.StateAxes({nnx.Variable:0, ...: None}), 0)
@nnx.vmap(in_axes=(0,0,0, None, 0,0, None)) # model params
def _train_step(graphdef, state, metric_split, data, optimizer_states, lr,criterion):
  model = nnx.merge(graphdef, state)
  metrics = nnx.merge(*metric_split)
  X, y = data
  def loss_fn(model):
    y_pred = model(X)[..., -1, :]
    return criterion(y_pred, y), y_pred
  (loss,logits), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
  metrics.update(logits=logits, loss=loss, labels=y)
  temp = nnx.Optimizer(model, optax.adamw(lr), wrt=nnx.Param)
  optimizer = nnx.merge(nnx.graphdef(temp), optimizer_states)
  optimizer.update(model, grads)
  return model, nnx.state(optimizer), metrics

def train_step(models, data, optimizer_states, config, metrics):
  models.train()
  graphdef, state = nnx.split(models)
  models, optimizer_states, metrics = _train_step(graphdef, state, nnx.split(metrics)  , data, optimizer_states, config.lrs, config.criterion)
  # models = nnx.merge(graphdef, state)
  # metrics.update(logits=logits, loss=loss, labels=y)
  return models, optimizer_states, metrics

@partial(jax.jit, static_argnames=('criterion'))
@nnx.vmap(in_axes=(0,0, None, None)) # data seeds #in_axes=(state_axes, nnx.StateAxes({nnx.Variable:0, ...: None}), 0)
@nnx.vmap(in_axes=(0,0, None, None)) # model params
# @partial(jax.jit, static_argnames=('criterion', 'data'))
def _eval_step(model, metric, data, criterion):
  # model, metrics = nnx.merge(graphdef, state)
  model = nnx.merge(*model)
  metric = nnx.merge(*metric)
  X, y = data
  logits = model(X)[..., -1, :]
  loss = criterion(logits, y)
  metric.update(logits=logits, loss=loss, labels=y)
  # (logits=logits, loss=loss, labels=y)
  return metric

def eval_step(models, data, config, metrics):
  models.eval()
  # graphdef, state = 
  metrics = _eval_step(nnx.split(models), nnx.split(metrics), data, config.criterion)
  # metrics = nnx.merge(graphdef, state)
  return metrics

@nnx.jit
@nnx.vmap
@nnx.vmap
def reset(metrics):
    metrics.reset()

def create_batch(rng, n: int, d: int, k: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Returns:
      X_train : (n, d)
      y_train : (n,)
    """
    # rng = rng()
    X   = jax.random.bernoulli(rng, p=0.5, shape=(n, d)).astype(jnp.int32)
    y = evaluate_parity(X, k)
    return X, y

def evaluate_parity(x: jnp.ndarray, k: int = 2) -> jnp.ndarray:
    # return jnp.prod(x[..., :k], axis=-1, dtype=jnp.float32)
    return jnp.sum(x[..., :k], axis=-1)% 2
    # return nnx.one_hot(jnp.sum(x[..., :k], axis=-1)% 2, 2, dtype=jnp.int32)
@partial(jax.jit, static_argnums=(1,2,3))
# @nnx.split_rngs(splits=config.num_seeds)
@nnx.vmap(in_axes=(0,None, None, None))
def create_batches(rng, n: int, d: int, k: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    return create_batch(rng(), n, d, k)



batch_size = 32
d = 20
k = 6


model_config = TransformerConfig(
    vocab_size=2,
    max_len=d,
    embd_dim=16,
    num_heads=2,
    num_layers = 2
)

config = TrainConfig(
    lrs = jnp.geomspace(1e-4,1e-1,20),
    num_seeds=20,
    # criterion = lambda y_pred, y: optax.squared_error(y_pred.squeeze(-1), y).mean(),
    criterion = lambda y_pred, y: optax.softmax_cross_entropy_with_integer_labels(y_pred, y).mean(),
    model = Transformer,
    model_config = model_config
)
models, optimizer_states, metrics = init_model(config)


metrics_history = {
  'train_loss': [],
  'train_accuracy': [],
  'test_loss': [],
  'test_accuracy': [],
}
@nnx.jit
@nnx.vmap
def iterate_rngs(rngs):
  rngs()
num_steps = 2000
data_rng = nnx.Rngs(0)
backup = nnx.split_rngs(data_rng, splits=config.num_seeds)
iterate_rngs(data_rng)
# print(models.lm_head.kernel.value[1][0])
test_data = create_batch(nnx.Rngs(-2)(), min(100, 2**d), d,   k)
for step in tqdm(range(num_steps)):
    data = create_batches(data_rng, batch_size, d, k)
    iterate_rngs(data_rng)
    models, optimizer_states, metrics = train_step(models, data, optimizer_states, config, metrics)
    acc, loss = metrics.compute().values()
    metrics_history[f'train_loss'].append(loss.mean(axis=0).min()) # Record the metrics.
    metrics_history[f'train_accuracy'].append(acc.mean(axis=0).max()) # Record the metrics.
    # Reset the metrics for the test set.
    reset(metrics)
    metrics = eval_step(models, test_data, config, metrics)
    acc, loss = metrics.compute().values()
    metrics_history[f'test_loss'].append(loss.mean(axis=0).min()) # Record the metrics.
    metrics_history[f'test_accuracy'].append(jnp.mean(acc,axis=0).max()) # Record the metrics.
      # Reset the metrics for the test set.
    reset(metrics)