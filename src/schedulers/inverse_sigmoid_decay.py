import math


def inverse_sigmoid_decay(step_fraction: float, decay_rate: float) -> float:
    """Inverse-sigmoid decay over normalized training progress.

    step_fraction is the position in training in [0, 1] (e.g.
    global_step / total_steps). The curve goes from ~1 at step_fraction=0 to
    ~0 at step_fraction=1, crossing 0.5 at step_fraction=0.5. decay_rate
    controls the steepness of the transition; 10 is a reasonable default.
    """
    return 1.0 / (1.0 + math.exp(decay_rate * (step_fraction - 0.5)))
