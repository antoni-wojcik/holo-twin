"""
The single generic training loop. Data-source-agnostic: sim and
experimental training call this with different `batch_fn`s (see data.py).
Supports gradient accumulation via micro-batching, LR scheduling, gradient
clipping, per-parameter activity logging, and an optional callback for
progress plotting/checkpointing (kept separate so this function has no
knowledge of ExpIO / matplotlib).
"""
import numpy as np
import torch
from dataclasses import dataclass
from typing import Callable, List, Optional
from tqdm import tqdm
from src.twin.model import HoloSystem


@dataclass
class DataSource:
    """
    A data source for a training stage, consisting of a batch function and the total number of samples.

    Parameters
    ----------
    batch_fn : callable
        Function that returns a batch of data when called.
    total_samples : int
        Total number of samples in the dataset (used to determine how many
        batches to draw per epoch).
    """
    batch_fn: Callable[[int], tuple]
    total_samples: int

@dataclass
class LossSpec:
    """
    A named, fixed-form loss/metric for evaluation and cross-run comparison.

    fn must depend only on (I, T) -- no regularization, no run-specific
    weighting -- so the same LossSpec means the same thing regardless of
    what a given run is actually optimizing.
    """
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    name: str
@dataclass
class LossSuite:
    """
    Separates the actual training objective (which may include
    regularization, custom weighting, detaching, etc. -- and differs
    between runs we are trying to compare) from a FIXED panel of pure
    fidelity metrics evaluated read-only alongside training.

    Parameters
    ----------
    train_loss : callable(I, T) -> scalar tensor
        The actual objective passed to backward().
    eval_losses : list of LossSpec
        Fixed comparison panel, evaluated under no_grad, not backpropagated.
    eval_data : DataSource, optional
        If given, the eval panel is evaluated on a fixed held-out batch
        from this source each epoch, rather than on the randomly
        sampled training micro-batch.
    validation_data : DataSource, optional
        Held-out data (disjoint from train_data) that must never be
        trained on. If given, every spec in `eval_losses` is ALSO
        evaluated per-sample over the whole of `validation_data` once per
        epoch, logging (mean, standard error of the mean) across its
        samples -- see evaluate_loss_mean_sem(). This is what produces the
        validation curve alongside the training curve for convergence
        plots (see plot_convergence()); distinct from `eval_data` above,
        which evaluates the panel on a single held-out batch rather than
        tracking a per-epoch mean/SEM over the full set.
    """
    train_loss: LossSpec
    train_data: DataSource
    eval_losses: Optional[List[LossSpec]] = None
    eval_data: Optional[DataSource] = None
    validation_data: Optional[DataSource] = None


PLOT_WEIGHTED_GRADIENTS = False  # if True, multiply each parameter's mean-abs-grad by its LR before logging

def train_model(
    system: HoloSystem,
    optimizer: torch.optim.Optimizer,
    loss_suite: LossSuite,
    num_epochs: int,
    micro_batch_size: int = 2,
    batch_size: int = 4,
    scheduler=None,
    grad_clip_norm: float = 1.0,
    callback=None, # callback(epoch, I_true, I_twin, history[:epoch+1] (each array sliced), activity_log)
    callback_every: int = 50,
    name: str = "Twin Training",
):
    """
    Parameters
    ----------
    system      : HoloSystem being trained
    loss_func   : callable(I_pred, I_true) -> scalar
    batch_fn    : idx (LongTensor) -> (g, I_true), see data.py
    total_samples : size of the (real or virtual) dataset to draw indices from
    num_epochs  : number of epochs to train for
    micro_batch_size : number of samples to process before calling backward()
    batch_size  : number of samples to accumulate gradients over before calling optimizer.step()
    scheduler   : optional LR scheduler (called once per epoch)
    grad_clip_norm : max norm for gradient clipping (prevents exploding gradients)
    callback    : optional hook invoked every `callback_every` epochs (and at
                  epoch 0 and the last epoch) for saving/plotting progress.
                  Keeping this generic means the same loop works whether
                  you're using ExpIO (experimental) or plt.show (sim).
    callback_every : how often to call the callback (in epochs)
    name        : string for progress bar title

    Returns
    -------
    history : dict
        {name: np.ndarray of shape (num_epochs,)}, keyed "train: <loss_suite.train_loss.name>"
        (the training objective, with reg), "eval: <spec.name>" for each spec in
        loss_suite.eval_losses (evaluated on the training micro-batch, no reg), and --
        if loss_suite.validation_data is set -- "validation: <spec.name> (mean)" / "(sem)"
        for each eval spec, evaluated per-sample over the whole of validation_data once per
        epoch (see evaluate_loss_mean_sem()). Built and filled in place, so this IS the final,
        self-describing, plot/save-ready form -- there's no separate array-to-dict conversion
        step; TrainingStage._report() in stage.py saves and plots it directly.
    activity_log : dict of lists, keyed by parameter name, each list contains
                   the mean absolute gradient of that parameter at each epoch
                   to track which parameters are "active" during training.
    """
    accum_steps = batch_size // micro_batch_size
    if batch_size % micro_batch_size != 0:
        eff = accum_steps * micro_batch_size
        print(f"Warning: batch_size {batch_size} not divisible by micro_batch_size "
              f"{micro_batch_size}; using effective batch size {eff}.")

    train_key = f"train: {loss_suite.train_loss.name if loss_suite.train_loss is not None else 'train'}"
    eval_keys = [] if loss_suite.eval_losses is None else [f"eval: {spec.name}" for spec in loss_suite.eval_losses]
    # One (mean, sem) pair per eval spec, only if there's somewhere held-out to evaluate them on.
    track_validation = bool(eval_keys) and loss_suite.validation_data is not None
    val_keys = []
    if track_validation:
        for spec in loss_suite.eval_losses:
            val_keys += [
                f"validation: {spec.name} (mean)", f"validation: {spec.name} (sem)",
                f"validation: {spec.name} (std)",
                f"validation: {spec.name} (min)", f"validation: {spec.name} (max)",
            ]

    history = {key: np.empty(num_epochs, dtype=np.float32) for key in [train_key, *eval_keys, *val_keys]}
    activity_log = {p_name: [] for p_name, _ in system.named_parameters()}

    bar = tqdm(range(num_epochs), desc=f"Training {name}", unit="epoch")
    for epoch in bar:
        optimizer.zero_grad()
        perm = torch.randperm(loss_suite.train_data.total_samples)

        epoch_loss_sum = 0.0
        epoch_eval_sum = (
            np.zeros(len(eval_keys), dtype=np.float32)
            if loss_suite.eval_losses is not None else None
        )

        I_true = I_twin = None  # keep last micro-batch around for the callback

        for i in range(accum_steps):
            idx = perm[i * micro_batch_size:(i + 1) * micro_batch_size]
            g, I_true = loss_suite.train_data.batch_fn(idx)

            I_twin = system(g)
            loss = loss_suite.train_loss.fn(I_twin, I_true)

            if loss_suite.eval_losses is not None:
                with torch.no_grad():
                    eval = [loss_spec.fn(I_twin, I_true).item() for loss_spec in loss_suite.eval_losses]

            epoch_loss_sum += loss.item()
            if epoch_eval_sum is not None:
                epoch_eval_sum += np.array(eval)

            bar.set_postfix({
                "iter": i + 1,
                "loss": f"{epoch_loss_sum / (i + 1):.4e}",
                "lr": ", ".join(f"{g['lr']:.2e}" for g in optimizer.param_groups),
            })

            (loss / accum_steps).backward()

        # Gradient clipping
        for group in optimizer.param_groups:
            torch.nn.utils.clip_grad_norm_(group["params"], max_norm=grad_clip_norm)

        with torch.no_grad():
            if PLOT_WEIGHTED_GRADIENTS:
                lr_map = _param_lr_map(optimizer)
            for p_name, p in system.named_parameters():
                if p.grad is not None:
                    act = p.grad.abs().mean()
                    if PLOT_WEIGHTED_GRADIENTS:
                        lr = lr_map.get(id(p), 0.0)
                        act = act * lr
                    activity_log[p_name].append(act.cpu().detach().item())

        history[train_key][epoch] = epoch_loss_sum / accum_steps
        if epoch_eval_sum is not None:
            for eval_key, val in zip(eval_keys, epoch_eval_sum / accum_steps):
                history[eval_key][epoch] = val

        # Held-out validation -- evaluated BEFORE optimizer.step() below
        if track_validation:
            for spec in loss_suite.eval_losses:
                mean, sem, std, vmin, vmax = evaluate_loss_mean_sem(
                    system, loss_suite.validation_data, spec,
                    micro_batch_size=micro_batch_size,
                )
                history[f"validation: {spec.name} (mean)"][epoch] = mean
                history[f"validation: {spec.name} (sem)"][epoch] = sem
                history[f"validation: {spec.name} (std)"][epoch] = std
                history[f"validation: {spec.name} (min)"][epoch] = vmin
                history[f"validation: {spec.name} (max)"][epoch] = vmax

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if callback is not None and (epoch == 0 or epoch % callback_every == 0 or epoch == num_epochs - 1):
            callback(epoch, I_true, I_twin, {k: v[:epoch + 1] for k, v in history.items()}, activity_log)

    return history, activity_log


def _param_lr_map(optimizer: torch.optim.Optimizer) -> dict:
    """
    id(param) -> the lr of whichever optimizer param group it belongs to.
    Needed because named_parameters() has no idea which group (and
    therefore which lr) a given parameter is in -- that's the whole point
    of having multiple param groups with different schedules.
    """
    return {id(p): group["lr"] for group in optimizer.param_groups for p in group["params"]}


def evaluate_losses(
    system: HoloSystem, data: DataSource, eval_losses: List[LossSpec],
    batch_size: int = 4, micro_batch_size: Optional[int] = None,
) -> dict:
    """
    Evaluate every LossSpec in `eval_losses` over the ENTIRE `data` source
    (sample-weighted average), under no_grad -- nothing here is trained
    on or backpropagated. Used for held-out validation, where the point
    is a single honest number per loss rather than a per-epoch training
    curve (see train_model()'s returned history dict for that).

    Parameters
    ----------
    system : HoloSystem
        The (already-trained) model to evaluate.
    data : DataSource
        Held-out data -- must not be the same data used for training.
    eval_losses : list of LossSpec
        The fixed comparison panel to evaluate.
    batch_size : int
        Total number of samples an average is computed over per call site
        -- kept for signature symmetry with training; does not affect how
        the model is actually forwarded (see micro_batch_size).
    micro_batch_size : int, optional
        Chunk size the model is actually forwarded in (defaults to
        batch_size, i.e. no micro-batching). no_grad avoids the backward
        pass's activation memory, but the forward pass itself can still
        be memory-bound at full batch_size -- that's exactly why training
        micro-batches in the first place (see train_model()) -- so this
        should normally be passed the same micro_batch_size used for
        training, not left at batch_size.

    Returns
    -------
    dict
        {loss_name: float}, one entry per LossSpec in `eval_losses`.
    """
    micro_batch_size = micro_batch_size or batch_size
    totals = {spec.name: 0.0 for spec in eval_losses}
    seen = 0
    with torch.no_grad():
        for start in range(0, data.total_samples, micro_batch_size):
            idx = torch.arange(start, min(start + micro_batch_size, data.total_samples))
            g, I_true = data.batch_fn(idx)
            I_pred = system(g)
            for spec in eval_losses:
                totals[spec.name] += spec.fn(I_pred, I_true).item() * len(idx)
            seen += len(idx)
    return {name: total / seen for name, total in totals.items()}


def evaluate_loss_mean_sem(
    system: HoloSystem, data: DataSource, loss_spec: LossSpec,
    micro_batch_size: int,
) -> tuple:
    """
    Evaluate `loss_spec` per-INDIVIDUAL-sample over the entire `data`
    source, returning (mean, sem, std, min, max) across data.total_samples
    samples. SEM captures how precisely the mean is known given only this
    many held-out samples; std/min/max capture the actual spread across
    those samples -- a different, generally much larger, quantity. See
    plot_convergence() for how mean/sem/min/max are used together (std is
    computed here but not plotted by default).
    """
    values = []
    with torch.no_grad():
        for start in range(0, data.total_samples, micro_batch_size):
            idx = torch.arange(start, min(start + micro_batch_size, data.total_samples))
            g, I_true = data.batch_fn(idx)
            I_pred = system(g)
            for i in range(len(idx)):
                values.append(loss_spec.fn(I_pred[i:i + 1], I_true[i:i + 1]).item())

    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    sem = std / np.sqrt(len(values)) if len(values) > 1 else 0.0
    return mean, sem, std, float(values.min()), float(values.max())