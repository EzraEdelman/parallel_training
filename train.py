from eq_parallel import init_models, train_step, eval_step, TrainConfig
from models.transformer import Transformer
from datasets.parity import Parity
from tqdm.auto import tqdm
import jax.numpy as jnp
import optax
import jax
import jax.random as jr
import matplotlib.pyplot as plt

def train(train_config: TrainConfig):
    num_steps = train_config.num_steps
    rng_key = train_config.rng_key
    rng_key, test_key = jr.split(rng_key)
    dataset = train_config.dataset(train_config.dataset_config)
    test_data = dataset.create_batch(test_key, 256)

    if train_config.trainset_size == -1:
        @jax.jit
        def get_data(key):
            return dataset.create_batches(key, train_config.batch_size, train_config.num_seeds)
    else:
        rng_key, data_key = jr.split(rng_key)
        offline_data = dataset.create_batches(data_key, train_config.trainset_size, train_config.num_seeds)
        @jax.vmap
        def _get_data(key, _offline_dataset):
            inds = jax.random.randint(key, (train_config.batch_size), 0, train_config.trainset_size,)
            return _offline_dataset[0][inds],_offline_dataset[1][inds]
        @jax.jit
        def get_data(key):
            return _get_data(jr.split(key, train_config.num_seeds), offline_data)
    
    rng_key, init_key = jr.split(rng_key)
    models, optimizer_states = init_models(train_config, init_key)

    metrics_history = {
        'train_loss': [],
        'train_accuracy': [],
        'test_loss': [],
        'test_accuracy': [],
    }
    pbar = tqdm(range(num_steps))
    for step in pbar:
        # Create training batches
        rng_key, batch_key, train_key = jr.split(rng_key, 3)
        
        data = get_data(batch_key)
        # Training step - returns models, opt_states, and train_metrics
        models, optimizer_states, train_metrics = train_step(
            models, optimizer_states, data, train_config.lrs, train_config.criterion, train_key, train_config.num_seeds
        )
        
        # Extract train metrics (shape: [num_seeds, num_lrs])
        train_acc = jnp.array(train_metrics['accuracy'])
        train_loss = jnp.array(train_metrics['loss'])
        
        # Record best metrics: mean across seeds, then min loss / max accuracy across LRs
        metrics_history['train_loss'].append(train_loss.mean(axis=0).min())
        metrics_history['train_accuracy'].append(train_acc.mean(axis=0).max())
        
        if step % 50 == 0:
            # Eval step - returns test_metrics
            test_metrics = eval_step(models, test_data, train_config.criterion)
            # Extract test metrics (shape: [num_seeds, num_lrs])
            test_acc = jnp.array(test_metrics['accuracy'])
            test_loss = jnp.array(test_metrics['loss'])
            # Record best metrics
            metrics_history['test_loss'].append(test_loss.mean(axis=0).min())
            metrics_history['test_accuracy'].append(test_acc.mean(axis=0).max())
            pbar.set_postfix(train_acc=f"{metrics_history['train_accuracy'][-1]:.2f}", test_acc=f"{metrics_history['test_accuracy'][-1]:.2f}")
    return metrics_history



if __name__ == "__main__":
    dataset_config = Parity.config(d=20, k=6, zero_one=True)

    # model_config = Transformer.config(
    #     vocab_size=2,
    #     max_len=dataset_config.d,
    #     embd_dim=256,
    #     mlp_dim=1024,
    #     qkv_dim=512,
    #     num_heads=8,
    #     num_layers=2,
    #     dtype=jnp.float32
    # )
    model_config = Transformer.config(
        vocab_size=2,
        max_len=dataset_config.d,
        embd_dim=256,
        mlp_dim=1024,
        qkv_dim=256,
        num_heads=8,
        num_layers=2,
        dtype=jnp.float32
    )

    config = TrainConfig(
        lrs=jnp.geomspace(1e-4, 1e-1, 10),
        num_seeds=10,
        criterion=lambda y_pred, y: optax.softmax_cross_entropy_with_integer_labels(y_pred, y).mean(),
        model=Transformer,
        trainset_size = -1,
        dataset=Parity,
        dataset_config=dataset_config,
        model_config=model_config,
        num_steps = 5000,
    )

    online_metrics_history = train(config)
    offline_config = config._replace(trainset_size=5000)
    offline_metrics_history = train(offline_config)

    metrics = {"online": online_metrics_history, "offline": offline_metrics_history, }
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.set_title('Loss')
    ax2.set_title('Accuracy')
    for metric in metrics:
        for dataset in ('test',):
            ax1.plot(metrics[metric][f'{dataset}_loss'], label=f'{metric} {dataset}_loss')
            ax2.plot(metrics[metric][f'{dataset}_accuracy'], label=f'{metric} {dataset}_accuracy')
    ax1.legend()
    ax2.legend()
    plt.savefig("parity_transformer_results.png")
