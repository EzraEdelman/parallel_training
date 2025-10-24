from flax import nnx  
import jax
import dataclasses

class Block(nnx.Module):
  def __init__(self, input_dim, features, rngs, kernel_init=None, bias_init=None):
    self.linear = nnx.Linear(input_dim, features, rngs=rngs, kernel_init=kernel_init, bias_init=bias_init)

  def __call__(self, x: jax.Array):  # No need to require a second input!
    x = self.linear(x)
    x = nnx.relu(x)
    return x   # No need to return a second output!


class MLP(nnx.Module):
  """
  MLP with ReLU activation. must have at least one hidden layer.
  """
  def __init__(self, config, rngs):
    num_hidden_layers = config.hidden_layers
    hidden_dim = config.hidden_dim
    in_dim = config.in_dim
    out_dim = config.out_dim
    if num_hidden_layers == 0:
      raise NotImplementedError("Number of hidden layers must be at least one.")
    @nnx.split_rngs(splits=num_hidden_layers-1)
    @nnx.vmap(in_axes=(0,), out_axes=0)
    def create_block(rngs: nnx.Rngs):
      return Block(hidden_dim, hidden_dim, rngs=rngs, kernel_init=config.kernel_init, bias_init=config.bias_init)
    self.input_block = Block(in_dim, hidden_dim, rngs=rngs, kernel_init=config.kernel_init, bias_init=config.bias_init)
    self.middle_blocks = create_block(rngs)
    self.output_block = Block(hidden_dim, out_dim, rngs=rngs, kernel_init=config.kernel_init, bias_init=config.bias_init)
    self.num_layers = num_hidden_layers

  def __call__(self, x):
    @nnx.split_rngs(splits=self.num_layers-1)
    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
    def forward(x, model):
      x = model(x)
      return x

    return self.output_block(forward(self.input_block(x), self.middle_blocks))
  
  @staticmethod
  def config(**kwargs):
    return MLPConfig(**kwargs)
  
@dataclasses.dataclass(unsafe_hash=True)
class MLPConfig:
  in_dim: int
  hidden_dim: int = 16
  out_dim: int = 1
  hidden_layers: int = 2
  kernel_init: nnx.Initializer = nnx.initializers.lecun_normal()
  bias_init: nnx.Initializer = nnx.initializers.zeros_init()

  def replace(self, **kwargs):
    return dataclasses.replace(self, **kwargs)