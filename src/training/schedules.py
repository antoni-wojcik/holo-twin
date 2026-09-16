"""
LR schedule factory. Every stage in the sim/experimental scripts uses the 
same warmup + cosine-decay shape, just with different iteration counts, and
occasionally a different warmup fraction / an early end point where the
LR flatlines at 0 (e.g. freezing the camera group partway through fine
optics training). One factory covers all of these cases.
"""
import numpy as np
import torch
 
 
def cosine_warmup_lambda(iterations: int, warmup_frac: float = 0.5,
                          decay_end_frac: float = 1.0, floor: float = 0.0):
    """
    Returns a callable `epoch -> lr multiplier` with three phases:
 
      1. flat 1.0                               for epoch < warmup_frac * iterations
      2. cosine decay from 1.0 down to `floor`  until epoch = decay_end_frac * iterations
      3. flat `floor`                           for all epochs after that
 
    Examples:
      - plain warmup+cosine-to-zero over the whole run:  (0.5, 1.0, 0.0)
      - "kill" a param group early (camera group during
        fine optics training, warmup 1/4, dead by half):  (0.25, 0.5, 0.0)

    Parameters
    ----------
    iterations : int
        Total number of epochs in the training run.
    warmup_frac : float
        Fraction of the run to keep LR flat at 1.0 before starting cosine decay.
    decay_end_frac : float
        Fraction of the run to end the cosine decay at `floor`.
    floor : float
        Final LR multiplier after decay ends (typically 0.0, but can be >0 for a "floor" LR).

    Returns
    -------
    lr_lambda : callable
        A function that takes an epoch index and returns the LR multiplier for that epoch.
    """
    warmup_epochs = int(iterations * warmup_frac)
    decay_end_epoch = int(iterations * decay_end_frac)
 
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return 1.0
        if epoch >= decay_end_epoch:
            return floor
        t = (epoch - warmup_epochs) / max(decay_end_epoch - warmup_epochs, 1)
        return floor + (1.0 - floor) * 0.5 * (1 + np.cos(np.pi * t))
 
    return lr_lambda
 
 
def make_scheduler(optimizer: torch.optim.Optimizer, iterations: int, schedules):
    """
    Build a LambdaLR scheduler for `torch.optim.Optimizer`.
 
    Example (two param groups: optics + camera, different decay for each):
        scheduler = make_scheduler(
            optimizer, iterations,
            [(0.5, 1.0, 0.0),   # optics group
             (0.25, 0.5, 0.0)], # camera group: shorter warmup, dead by halfway
        )
    
    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer to attach the scheduler to.
    iterations : int
        Total number of epochs in the training run.
    schedules : list of tuples or single tuple
        - a single (warmup_frac, decay_end_frac, floor) tuple, applied to
        every param group, or
        - a list of such tuples, one per param group (must match
        len(optimizer.param_groups)).

    Returns
    -------
    scheduler : torch.optim.lr_scheduler.LambdaLR
        The LR scheduler that can be stepped each epoch.
    """
    n_groups = len(optimizer.param_groups)
    if isinstance(schedules, tuple):
        schedules = [schedules] * n_groups
    if len(schedules) != n_groups:
        raise ValueError(f"Got {len(schedules)} schedules for {n_groups} param groups.")
 
    lambdas = [cosine_warmup_lambda(iterations, *s) for s in schedules]
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambdas if n_groups > 1 else lambdas[0]
    )