def exponential_decay(epsilon: float, step: int, decay_rate: float) -> float:
    """Exponentially decay epsilon with a lower bound."""
    return max(epsilon, decay_rate**step)
