# Taylor Series in Active Inference

## Overview

Taylor series expansion plays a crucial role in continuous-time active inference by allowing us to:
1. Represent continuous trajectories using discrete samples
2. Connect generalized coordinates to predictions
3. Formalize the relationship between different orders of motion

## Mathematical Foundation

### Basic Definition

The Taylor series expansion of a function f(t) around t₀ is:

```
f(t) = f(t₀) + f'(t₀)(t-t₀) + f''(t₀)(t-t₀)²/2! + f'''(t₀)(t-t₀)³/3! + ...
```

In generalized coordinates, this becomes:

```
f(t) = x + x'(t-t₀) + x''(t-t₀)²/2! + x'''(t-t₀)³/3! + ...
```

where [x, x', x'', x'''] are the generalized coordinates.

## Role in Active Inference

### 1. Predictive Processing

Taylor series enables:
- Forward predictions in time
- Smooth interpolation between time points
- Uncertainty propagation across orders

### 2. Belief Updating

The relationship between orders is formalized through Taylor series:
```
dx/dt = x'
dx'/dt = x''
dx''/dt = x'''
```

### 3. Error Minimization

Prediction errors at each order contribute to free energy:
```
ε₀ = y - g(x)           # Sensory prediction error
ε₁ = x' - f(x)          # Motion prediction error
ε₂ = x'' - f'(x)        # Acceleration prediction error
```

## Implementation Details

### 1. Expansion in Code

```python
def taylor_expansion(x0, orders, time_points):
    expansion = np.zeros_like(time_points)
    for n in range(orders + 1):
        expansion += (x0[n] / factorial(n)) * (time_points)**n
    return expansion
```

### 2. Connection to Shift Operator

The shift operator D implements the Taylor series relationship:
```python
def create_shift_operator(n_orders):
    D = np.zeros((n_orders, n_orders))
    for i in range(n_orders - 1):
        D[i, i+1] = factorial(i+1) / factorial(i)
    return D
```

### 3. Prediction Generation

```python
def predict_future(current_state, dt):
    prediction = np.zeros_like(current_state)
    for n in range(len(current_state)):
        prediction += current_state[n] * dt**n / factorial(n)
    return prediction
```

## Advantages

1. **Smooth Predictions**: Natural interpolation between time points
2. **Error Hierarchy**: Structured prediction errors across orders
3. **Temporal Consistency**: Enforced through series relationships
4. **Computational Efficiency**: Truncated series for approximation

## Visualization

1. **Expansion Accuracy**: Compare different orders of expansion
2. **Prediction Quality**: Show how higher orders improve predictions
3. **Error Analysis**: Visualize errors at different orders
4. **Convergence**: Show how adding terms improves accuracy

## Example Analysis

### 1. Expansion Accuracy

```python
# Plot different orders of Taylor expansion
orders = [1, 2, 3, 4]
time_points = np.linspace(0, 1, 100)
x0 = get_initial_state()  # [x, x', x'', x''']

for order in orders:
    expansion = taylor_expansion(x0, order, time_points)
    plt.plot(time_points, expansion, label=f'Order {order}')
```

### 2. Error Analysis

```python
# Compare prediction errors at different orders
for order in range(n_orders):
    errors = compute_prediction_errors(order)
    plt.subplot(n_orders, 1, order+1)
    plt.plot(time_points, errors)
    plt.title(f'Order {order} Errors')
```

## References

1. Friston, K. J. (2008). Hierarchical models in the brain.
2. Buckley, C. L., et al. (2017). The free energy principle for action and perception: A mathematical review.
3. Baltieri, M., & Buckley, C. L. (2019). PID control as a process of active inference with linear generative models. 