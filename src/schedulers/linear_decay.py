def linear_decay(epsilon: float, step: int, offset: float, slope: float) -> float:
    """Linearly decay epsilon with a lower bound."""
    return max(epsilon, offset - slope * step)
