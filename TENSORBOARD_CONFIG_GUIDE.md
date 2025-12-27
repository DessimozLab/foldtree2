# Learn MonoDecoder - Enhanced Features

This document describes the enhanced features added to the `learn_monodecoder.py` training script.

## New Features

### 1. Configuration File Support

You can now use YAML or JSON configuration files to specify all training parameters. This makes it easier to manage different training runs and share configurations.

#### Usage:

```bash
# Using a config file
python learn_monodecoder.py --config my_config.yaml

# Override specific values from config file with CLI arguments
python learn_monodecoder.py --config my_config.yaml --learning-rate 0.0005 --epochs 50
```

#### Creating a Config File:

You can save your current configuration to a file:

```bash
python learn_monodecoder.py --dataset my_data.h5 --learning-rate 0.0001 --epochs 100 --save-config my_training.yaml
```

This will create a YAML file with all the current settings, which you can then edit and reuse.

#### Example Config File:

See `example_config.yaml` for a complete example with all available parameters.

```yaml
# Basic training configuration
dataset: structs_traininffttest.h5
batch_size: 20
hidden_size: 256
embedding_dim: 128
epochs: 100
learning_rate: 0.0001
model_name: my_experiment
tensorboard_dir: ./runs/
```

### 2. Enhanced TensorBoard Support

The script now includes comprehensive TensorBoard logging with better organization.

#### Features:

- **Organized Log Directories**: Each training run gets its own timestamped directory
- **Custom Run Names**: Specify a memorable name for your run
- **Comprehensive Metrics Tracking**:
  - All loss components (AA, Edge, VQ, FFT2, Angles, Logit)
  - Total loss
  - Learning rate over time
  - Loss weights over time
  - Gradient norms (min/max for encoder and decoder)
- **Hyperparameter Logging**: All hyperparameters are saved for easy comparison
- **Configuration Text**: Full configuration visible in TensorBoard

#### Usage:

```bash
# Basic usage (auto-generated run name with timestamp)
python learn_monodecoder.py --config my_config.yaml

# Custom run name
python learn_monodecoder.py --config my_config.yaml --run-name my_experiment_v1

# Custom TensorBoard directory
python learn_monodecoder.py --config my_config.yaml --tensorboard-dir ./my_runs/
```

#### Viewing TensorBoard:

```bash
# Start TensorBoard server
tensorboard --logdir=./runs/

# Then open your browser to http://localhost:6006
```

#### What You'll See in TensorBoard:

1. **Scalars Tab**:
   - Loss/AA, Loss/Edge, Loss/VQ, Loss/FFT2, Loss/Angles, Loss/Logit, Loss/Total
   - Learning_Rate
   - Weights/Edge, Weights/X, Weights/FFT2, Weights/VQ
   - Gradients/Encoder_Max, Gradients/Encoder_Min, Gradients/Decoder_Max, Gradients/Decoder_Min

2. **HParams Tab**:
   - Compare different runs based on hyperparameters
   - View metrics (best_loss, final_aa_loss, final_edge_loss)

3. **Text Tab**:
   - Full configuration for the run

## Complete Example Workflow

### 1. Create and Save a Configuration:

```bash
python learn_monodecoder.py \
  --dataset structs_traininffttest.h5 \
  --hidden-size 256 \
  --embedding-dim 128 \
  --epochs 100 \
  --learning-rate 0.0001 \
  --batch-size 20 \
  --output-dir ./models/ \
  --model-name my_model \
  --save-config baseline_config.yaml
```

### 2. Train with the Configuration:

```bash
python learn_monodecoder.py \
  --config baseline_config.yaml \
  --run-name baseline_experiment
```

### 3. Experiment with Different Parameters:

```bash
# Higher learning rate experiment
python learn_monodecoder.py \
  --config baseline_config.yaml \
  --learning-rate 0.0005 \
  --run-name high_lr_experiment

# Different architecture experiment
python learn_monodecoder.py \
  --config baseline_config.yaml \
  --hidden-size 512 \
  --run-name large_model_experiment
```

### 4. View Results in TensorBoard:

```bash
tensorboard --logdir=./runs/
```

Compare all your experiments side-by-side in TensorBoard's interface.

## Configuration File Parameters

All command-line arguments can be specified in the config file:

- `dataset`: Path to dataset file
- `hidden_size`: Hidden layer size
- `epochs`: Number of training epochs
- `device`: Device to use (cuda:0, cuda:1, cpu, or null for auto)
- `learning_rate`: Learning rate
- `batch_size`: Batch size
- `output_dir`: Directory to save models
- `model_name`: Name for the model
- `num_embeddings`: Number of embeddings for encoder
- `embedding_dim`: Embedding dimension
- `se3_transformer`: Use SE3Transformer (true/false)
- `overwrite`: Overwrite existing models (true/false)
- `output_fft`: Train with FFT output (true/false)
- `output_rt`: Train with rotation/translation output (true/false)
- `output_foldx`: Train with Foldx energy prediction (true/false)
- `seed`: Random seed for reproducibility
- `hetero_gae`: Use HeteroGAE_Decoder (true/false)
- `clip_grad`: Enable gradient clipping (true/false)
- `burn_in`: Burn-in period for training
- `EMA`: Use Exponential Moving Average (true/false)
- `tensorboard_dir`: Directory for TensorBoard logs
- `run_name`: Name for this training run (null for auto-generated)
- `lr_schedule`: Learning rate schedule ('plateau', 'cosine', 'linear', 'none')
- `lr_warmup_steps`: Number of steps for linear LR warmup (0 for no warmup)
- `lr_min`: Minimum learning rate for cosine/linear schedules

## Learning Rate Scheduling

The training script now supports advanced learning rate scheduling with linear warmup:

### Available Schedules

1. **`plateau`** (default): ReduceLROnPlateau - reduces LR when loss plateaus
2. **`cosine`**: Cosine annealing decay
3. **`linear`**: Linear decay from initial LR to minimum LR
4. **`none`**: No scheduling, constant learning rate

### Linear Warmup

All schedules support an optional linear warmup phase:
- Set `lr_warmup_steps` to the number of training steps for warmup
- Learning rate linearly increases from 0 to the target `learning_rate`
- After warmup, the specified schedule takes over

### Examples

**Cosine schedule with warmup:**
```yaml
learning_rate: 0.0001
lr_schedule: cosine
lr_warmup_steps: 1000  # Warmup for first 1000 steps
lr_min: 1.0e-06        # Minimum LR at end of training
```

**Plateau-based (default, adaptive):**
```yaml
learning_rate: 0.0001
lr_schedule: plateau
lr_warmup_steps: 500   # Optional warmup
```

**Linear decay with warmup:**
```yaml
learning_rate: 0.0001
lr_schedule: linear
lr_warmup_steps: 2000
lr_min: 1.0e-05
```

**No scheduling:**
```yaml
learning_rate: 0.0001
lr_schedule: none
lr_warmup_steps: 0
```

## Tips

1. **Use meaningful run names**: This makes it easier to identify experiments in TensorBoard
2. **Keep config files**: Store successful configurations for reproducibility
3. **Monitor gradients**: Watch the gradient norm plots to detect vanishing/exploding gradients
4. **Compare runs**: Use TensorBoard's HParams tab to find optimal hyperparameters
5. **Version control**: Keep your config files in git for full experiment tracking

## Troubleshooting

- If TensorBoard doesn't show your run, make sure the `tensorboard_dir` exists and check the console output for the log directory path
- Config file must be valid YAML or JSON - use a validator if you get parsing errors
- Command-line arguments always override config file values - use this for quick experiments
