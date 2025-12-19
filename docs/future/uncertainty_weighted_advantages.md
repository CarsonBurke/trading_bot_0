# Uncertainty-Weighted Advantages

## Concept
Use the distributional critic's entropy as a confidence signal to weight policy gradients. States where the critic is uncertain (high entropy distribution) should contribute less to policy updates.

## Implementation

```rust
// Compute critic distribution entropy
let critic_probs = critic_logits.softmax(-1, Kind::Float);
let critic_entropy = -(critic_probs * (critic_probs + 1e-8).log())
    .sum_dim_intlist(-1, false, Kind::Float);

// Convert entropy to confidence (inverse relationship)
// Normalize by max entropy (uniform distribution) for [0, 1] range
let max_entropy = (NUM_BUCKETS as f64).ln();
let normalized_entropy = &critic_entropy / max_entropy;
let confidence = 1.0 - normalized_entropy.clamp(0.0, 1.0);

// Weight advantages by confidence
// Low confidence → reduced gradient contribution
let weighted_advantages = advantages * confidence;
```

## Rationale
- High entropy in critic distribution indicates uncertain value estimates
- Policy gradients from uncertain states are unreliable
- Downweighting these reduces gradient variance
- Similar in spirit to PPG's careful value/policy separation

## Considerations
- May slow learning in novel states (where uncertainty is legitimate)
- Could add a floor to confidence (e.g., `confidence.clamp(0.3, 1.0)`) to avoid completely ignoring uncertain states
- Monitor critic entropy distribution to tune parameters

## Related Work
- Bootstrapped DQN (uncertainty via ensemble disagreement)
- Distributional RL (IQN, QR-DQN use full distribution)
- PPG (separates value learning to reduce interference)
