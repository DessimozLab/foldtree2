# Commitment Cost Scheduling for VectorQuantizerEMA

This document describes the commitment cost scheduling feature added to the `VectorQuantizerEMA` class in `foldtree2/src/quantizers.py`.

## Overview

The commitment cost is a crucial hyperparameter in vector quantization that controls the balance between:
- **Encoder commitment**: How much the encoder should commit to mapping inputs close to codebook entries
- **Codebook flexibility**: How much the codebook can adapt to the encoded representations

A **warmup schedule** for the commitment cost can improve training stability and final performance by:
1. Starting with a low commitment cost to allow the codebook to initialize properly
2. Gradually increasing it to the target value to encourage encoder commitment
3. Using a smooth schedule (cosine or linear) to avoid training instability

## New Parameters

### `use_commitment_scheduling` (default: False)
**Boolean flag to enable or disable commitment cost scheduling.**

- **True**: Enable scheduling with warmup from `commitment_start` to `commitment_end`
- **False**: Use constant `commitment_cost` throughout training (original behavior)

This flag makes it easy to turn scheduling on/off without changing other parameters.

### `commitment_warmup_steps` (default: 5000)
The number of training steps over which the commitment cost will be scheduled from `commitment_start` to `commitment_end`.

**Default**: 5000 steps is chosen as a reasonable default that:
- Allows sufficient warmup for most training scenarios
- Represents ~250 epochs with batch_size=20 on a 1000-sample dataset
- Can be adjusted based on your dataset size and training regime

### `commitment_schedule` (default: 'cosine')
The type of schedule to use for the commitment cost warmup:
- **'cosine'**: Smooth cosine annealing from start to end (recommended)
- **'linear'**: Linear interpolation from start to end
- **'none'**: No scheduling, use final value immediately

### `commitment_start` (default: 0.1)
The initial commitment cost value at the beginning of training.

**Why 0.1?** Starting with a lower value (compared to typical final values like 0.25-1.0) allows:
- Codebook to initialize without over-committing the encoder
- More exploration in the early training phase
- Smoother convergence

### `commitment_end` (default: None, uses `commitment_cost`)
The final commitment cost value after warmup completes. If not specified, uses the `commitment_cost` parameter.

## Usage Examples

### Default Behavior (No Scheduling - Original Behavior)

```python
from foldtree2.src.quantizers import VectorQuantizerEMA

# Default: scheduling is disabled, uses constant commitment cost
quantizer = VectorQuantizerEMA(
    num_embeddings=512,
    embedding_dim=128,
    commitment_cost=0.25,
    # use_commitment_scheduling=False,  # Default - scheduling disabled
)
```

### Enable Scheduling with Defaults

```python
# Enable scheduling with recommended defaults
quantizer = VectorQuantizerEMA(
    num_embeddings=512,
    embedding_dim=128,
    commitment_cost=0.25,  # This becomes commitment_end
    use_commitment_scheduling=True,  # Enable scheduling
    # commitment_warmup_steps=5000,  # Default
    # commitment_schedule='cosine',  # Default
    # commitment_start=0.1,  # Default
)
```

### Custom Warmup Schedule

```python
# Longer warmup with higher final commitment cost
quantizer = VectorQuantizerEMA(
    num_embeddings=512,
    embedding_dim=128,
    commitment_cost=0.5,
    use_commitment_scheduling=True,  # Enable scheduling
    commitment_warmup_steps=10000,  # Longer warmup
    commitment_schedule='cosine',
    commitment_start=0.05,  # Start even lower
)
```

### Linear Schedule

```python
# Linear interpolation instead of cosine
quantizer = VectorQuantizerEMA(
    num_embeddings=512,
    embedding_dim=128,
    commitment_cost=0.25,
    use_commitment_scheduling=True,  # Enable scheduling
    commitment_warmup_steps=5000,
    commitment_schedule='linear',
    commitment_start=0.1,
)
```

### Disable Scheduling (Original Behavior)

```python
# Two ways to disable scheduling:

# Method 1: Simply don't set the flag (default is False)
quantizer = VectorQuantizerEMA(
    num_embeddings=512,
    embedding_dim=128,
    commitment_cost=0.25,
)

# Method 2: Explicitly disable
quantizer = VectorQuantizerEMA(
    num_embeddings=512,
    embedding_dim=128,
    commitment_cost=0.25,
    use_commitment_scheduling=False,  # Explicitly disabled
)
```

### Custom Start and End Values

```python
# Fine control over start and end values
quantizer = VectorQuantizerEMA(
    num_embeddings=512,
    embedding_dim=128,
    commitment_cost=0.8,  # Ignored if commitment_end is set
    use_commitment_scheduling=True,  # Enable scheduling
    commitment_warmup_steps=8000,
    commitment_schedule='cosine',
    commitment_start=0.05,
    commitment_end=0.6,  # Explicitly set final value
)
```

## Monitoring During Training

### Get Current Commitment Cost

```python
# During training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        z, vq_loss = quantizer(batch)
        
        # Get current commitment cost for logging
        current_cost = quantizer.get_commitment_cost()
        
        # Log to tensorboard or print
        writer.add_scalar('VQ/commitment_cost', current_cost, step)
        print(f"Step {step}, Commitment Cost: {current_cost:.4f}")
```

### Reset Schedule (if needed)

```python
# Reset the schedule to start warmup from beginning
quantizer.reset_commitment_schedule()

# Useful if you want to:
# - Restart training with a new warmup
# - Switch phases in multi-stage training
```

## Integration with Training Scripts

### Example: learn_monodecoder.py

```python
# In encoder initialization
encoder = ft2.mk1_Encoder(
    in_channels=ndim,
    hidden_channels=[hidden_size, hidden_size],
    out_channels=args.embedding_dim,
    metadata={'edge_types': [('res','contactPoints','res')]},
    num_embeddings=args.num_embeddings,
    commitment_cost=0.9,  # Final target commitment cost
    # Enable scheduling
    use_commitment_scheduling=True,
    commitment_warmup_steps=5000,  # 5000 steps warmup
    commitment_schedule='cosine',  # Smooth cosine schedule
    commitment_start=0.1,  # Start at 10% of final value
    edge_dim=1,
    encoder_hidden=hidden_size,
    EMA=args.EMA,
    nheads=8,
    dropout_p=0.01,
    reset_codes=False,
    flavor='transformer',
    fftin=True
)

# In training loop - commitment cost updates automatically
for epoch in range(epochs):
    for batch in train_loader:
        z, vq_loss = encoder(batch)
        
        # Optional: Log commitment cost
        if step % 100 == 0:
            current_cost = encoder.vq_layer.get_commitment_cost()
            writer.add_scalar('VQ/commitment_cost', current_cost, step)
```

## Schedule Visualization

### Cosine Schedule (Recommended)
```
Commitment Cost
    |
1.0 |                    ▄▀▀▀▀▀▀▀▀▀▀
0.9 |                  ▄▀
0.8 |                ▄▀
0.7 |               ▀
0.6 |             ▄▀
0.5 |           ▄▀
0.4 |         ▄▀
0.3 |       ▄▀
0.2 |     ▄▀
0.1 |▄▀▀▀▀
    |_________________________
    0   1k  2k  3k  4k  5k steps
```

**Benefits of Cosine**:
- Faster initial increase for quick codebook initialization
- Slower approach to final value for stability
- Smooth gradients throughout

### Linear Schedule
```
Commitment Cost
    |
1.0 |                      ▄
0.9 |                    ▄▀
0.8 |                  ▄▀
0.7 |                ▄▀
0.6 |              ▄▀
0.5 |            ▄▀
0.4 |          ▄▀
0.3 |        ▄▀
0.2 |      ▄▀
0.1 |▄▄▄▄▀
    |_________________________
    0   1k  2k  3k  4k  5k steps
```

## Recommended Settings by Dataset Size

### Small Dataset (< 1000 samples)
```python
use_commitment_scheduling=True
commitment_warmup_steps=2000  # Shorter warmup
commitment_start=0.1
commitment_cost=0.25  # Moderate final value
commitment_schedule='cosine'
```

### Medium Dataset (1000-10000 samples)
```python
use_commitment_scheduling=True
commitment_warmup_steps=5000  # Default - good balance
commitment_start=0.1
commitment_cost=0.5  # Higher commitment
commitment_schedule='cosine'
```

### Large Dataset (> 10000 samples)
```python
use_commitment_scheduling=True
commitment_warmup_steps=10000  # Longer warmup
commitment_start=0.05  # Start lower
commitment_cost=0.8  # Strong commitment
commitment_schedule='cosine'
```

## Troubleshooting

### Codebook Collapse (Many Unused Codes)
- **Increase** `commitment_warmup_steps` (e.g., 10000)
- **Decrease** `commitment_start` (e.g., 0.05)
- Use `commitment_schedule='cosine'` for smoother warmup

### Poor Reconstruction Quality
- **Decrease** `commitment_start` to allow more encoder flexibility early on
- **Increase** final `commitment_cost` for stronger commitment
- Monitor commitment cost curves - ensure it reaches final value

### Training Instability
- **Increase** `commitment_warmup_steps` for more gradual change
- Use `commitment_schedule='cosine'` instead of 'linear'
- Start with lower `commitment_start` value

## Technical Details

### Schedule Formulas

**Cosine Schedule**:
```python
c = 0.5 * (1 + cos(π * t / T))
commitment_cost = end + (start - end) * c
```
where:
- `t` = current training step
- `T` = total warmup steps
- `start` = initial commitment cost
- `end` = final commitment cost

**Linear Schedule**:
```python
commitment_cost = start + (end - start) * (t / T)
```

### Automatic Updates

The commitment cost is updated automatically during the forward pass when both `self.training == True` and `self.use_commitment_scheduling == True`:

```python
def forward(self, x):
    if self.training and self.use_commitment_scheduling:
        self.update_commitment_cost()
        self.current_step += 1
    # ... rest of forward pass
```

This ensures the schedule progresses with each training batch only when scheduling is enabled.

## References

The commitment cost scheduling is inspired by:
- Warmup strategies in transformer training (learning rate warmup)
- Cosine annealing schedules for training stability
- Best practices in VQ-VAE training

## Backward Compatibility

The new parameters have sensible defaults that maintain **exact** original behavior:
- **`use_commitment_scheduling=False` by default**: Scheduling is opt-in, not enabled by default
- When scheduling is disabled, uses constant `commitment_cost` throughout training
- No changes to existing code needed - fully backward compatible

To enable the new scheduling feature, simply add:
```python
use_commitment_scheduling=True
```

All other scheduling parameters have reasonable defaults:
- Default warmup of 5000 steps
- Cosine schedule for smooth, stable training
- Starting at 0.1 allows proper initialization
