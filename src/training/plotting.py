"""
Diagnostic plotting shared across sim and experimental scripts. 
Every function here now BUILDS and RETURNS a matplotlib Figure instead of
calling plt.show() itself. That's what lets run_stage() decide, per call,
whether to display it interactively (sim, exploratory) or save it via
ExpIO and close it. Use `render()` to make that decision.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from src.twin.model import HoloSystem
from src.io.data_io import ExpIO

DPI = 200
 
 
def render(fig: plt.Figure, name: str = None, exp: ExpIO = None, subdir_name: str = None):
    """
    exp given  -> save the figure via ExpIO under subdir_name, then close it
                  (no interactive window; correct for unattended/real runs).
    exp is None -> plt.show() it (all currently-open figures are shown)
    """
    if exp is not None:
        exp.save_figure(fig, name or "figure", subdir_name=subdir_name)
        plt.close(fig)
    else:
        plt.show()

def render_image(img, cmap: str = 'gray', square: bool = False, subplot: bool = False, name: str = None, exp: ExpIO = None, subdir_name: str = None):
    """
    exp given  -> save the image via ExpIO under subdir_name
                  (no interactive window; correct for unattended/real runs).
    exp is None -> plt.show() it (all currently-open figures are shown)
    """

    data = img.detach().cpu().numpy() if torch.is_tensor(img) else img
    
    if exp is not None:
        exp.save_image_color(data, cmap=cmap, square=square, name=name or "image", subdir_name=subdir_name)
    else:
        show_image(data, title=name or "Image", cmap=cmap, square=square, subplot=subplot)
        if not subplot:
            plt.show()
 
 
def plot_image(img, title: str = "", cmap: str = "gray", colorbar: bool = True) -> plt.Figure:
    fig = plt.figure(dpi=DPI)
    data = img.detach().cpu().numpy() if torch.is_tensor(img) else img
    plt.imshow(data, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    if colorbar:
        plt.colorbar()
    return fig

 
def plot_loss(history: dict) -> plt.Figure:
    """
    Generic per-epoch loss plot: one line per entry in `history`, plotted
    under its own key as the label. `history` is exactly what
    train_model() in src/training/train.py returns (or, for
    CameraEstimateCVStage, whatever single-entry dict its own fit loop
    returns) -- no conversion step, no knowledge of LossSuite needed here.
    """
    fig = plt.figure(dpi=DPI)
    plt.title("Training Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    for key, values in history.items():
        plt.plot(values, label=key)
    plt.legend()
    plt.grid()
    return fig


def plot_convergence(history: dict, log_scale: bool = True) -> plt.Figure:
    """
    Per-stage convergence figure: training loss with and without the
    regularisation term (thin lines -- the gap between them is how much
    the regulariser is costing), and the held-out validation loss (the
    same no-reg quantity, "structure") as a mean +/- standard-error-of-
    the-mean band evaluated once per epoch. Called once per stage (optics,
    background) -- there is no combined end-of-run plot.

    Parameters
    ----------
    history : dict
        {name: np.ndarray of shape (epochs,)}, exactly what train_model()
        returns for this stage (see TrainingStage._report() in stage.py) --
        must contain "eval: structure" (the no-reg training objective) and
        exactly one other key with no "eval: "/"validation: " prefix (the
        with-reg training objective, whatever its LossSpec is named).
        "validation: structure (mean)"/"(sem)" are optional -- drawn as a
        shaded band only if present (i.e. loss_suite.validation_data was set).
    log_scale : bool
        Log-scale the y-axis (loss typically spans orders of magnitude
        early vs. late in training).
    """
    if "eval: structure" not in history:
        raise ValueError(
            "plot_convergence: history has no 'eval: structure' key -- see run_training() "
            "in src/training/pipeline.py, where the structure (no-reg) LossSpec should be "
            "named exactly 'structure'."
        )
    train_keys = [k for k in history if not k.startswith("eval: ") and not k.startswith("validation: ")]
    if len(train_keys) != 1:
        raise ValueError(
            f"plot_convergence: expected exactly one train-loss key (no 'eval: '/'validation: ' "
            f"prefix) in history, found {train_keys}."
        )
    train_key = train_keys[0]
    x = np.arange(len(history[train_key]))

    fig = plt.figure(dpi=DPI)
    plt.title("Convergence")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    if log_scale:
        plt.yscale("log")

    plt.plot(x, history[train_key], lw=1, alpha=0.8, color="C0", label="train (with reg)")
    plt.plot(x, history["eval: structure"], lw=1, alpha=0.8, color="C1", label="train (structure)")

    val_mean_key, val_sem_key = "validation: structure (mean)", "validation: structure (sem)"
    val_min_key, val_max_key = "validation: structure (min)", "validation: structure (max)"
    if val_mean_key in history:
        val_mean, val_sem = history[val_mean_key], history[val_sem_key]
        plt.plot(x, val_mean, lw=1, alpha=0.8, color="C2", label="validation (structure)")
        if val_min_key in history:   # older saved histories may not have min/max -- skip gracefully
            plt.fill_between(x, history[val_min_key], history[val_max_key],
                              color="C2", alpha=0.12, label="validation (min-max)")
        plt.fill_between(x, val_mean - val_sem, val_mean + val_sem, color="C2", alpha=0.35)
    plt.legend()
    plt.grid()
    return fig


def plot_cgh_loss(loss_history: np.ndarray, title: str = "Training Loss") -> plt.Figure:
    fig = plt.figure()
    plt.plot(loss_history)
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.grid()
    return fig
 
def plot_activity(activity_log: dict, title: str = "Parameter Activity") -> plt.Figure:
    fig = plt.figure()
    for p_name, activity in activity_log.items():
        activity = np.array(activity)
        if len(activity) > 0:
            activity = activity / (np.max(activity) + 1e-12)
            plt.plot(activity, label=p_name)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Normalized Mean Absolute Gradient")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    return fig
 
 
def _side_by_side(A: torch.Tensor, B: torch.Tensor, labels: tuple) -> plt.Figure:
    fig = plt.figure(figsize=(10, 4), dpi=DPI)
    for k, (im, label) in enumerate([(A, labels[0]), (B, labels[1])]):
        plt.subplot(1, 2, k + 1)
        plt.imshow(im.detach().cpu().numpy(), cmap='gray')
        plt.title(label)
        plt.colorbar()
        plt.axis('off')
    plt.tight_layout()
    return fig
 
 
def compare_camera_images(system_a: HoloSystem, system_b: HoloSystem, g: torch.Tensor, labels=("A", "B")) -> plt.Figure:
    """Two HoloSystems, same input -- used in simulation (twin vs ground truth)."""
    with torch.no_grad():
        C_a, C_b = system_a(g), system_b(g)
    return _side_by_side(C_a[0], C_b[0], labels)
 
 
def compare_prediction(model: HoloSystem, g: torch.Tensor, I_true: torch.Tensor, labels=("Measured", "Predicted")) -> plt.Figure:
    """One HoloSystem vs an already-measured image -- used in experimental
    training, where there's no second "true" model to forward through."""
    with torch.no_grad():
        I_pred = model(g)
    return _side_by_side(I_true[0], I_pred[0], labels)
 
 
def compare_slm_fields(system_a: HoloSystem, system_b: HoloSystem, labels: tuple = ("True", "Twin")) -> plt.Figure:
    fields = [system_a.slm_field(), system_b.slm_field()]
    fig = plt.figure(figsize=(10, 6))
    for col, (field, label) in enumerate(zip(fields, labels)):
        plt.subplot(2, 2, col + 1)
        plt.imshow(field.angle().detach().cpu().numpy(), cmap='twilight')
        plt.title(f"{label} Phase")
        plt.colorbar()
        plt.subplot(2, 2, col + 3)
        plt.imshow(field.abs().detach().cpu().numpy(), cmap='hot')
        plt.title(f"{label} Amplitude")
        plt.colorbar()
    plt.tight_layout()
    return fig

def show_image(img, title: str = "Image", cmap: str = 'gray', square: bool = False, subplot: bool = False):
    """
    Utility function to display a single image with a title and colorbar. 
    If `square` is True, the aspect ratio is set to equal. 
    If `subplot` is True, the image is plotted in the current subplot; 
    otherwise, a new figure is created.
    """
    data = img.detach().cpu().numpy() if torch.is_tensor(img) else img

    if not subplot:
        fig = plt.figure(dpi=DPI)
    if square:
        plt.imshow(data, cmap=cmap, extent=[0, 1, 0, 1], aspect='equal')
    else:
        plt.imshow(data, cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.axis('off')
    if not subplot:
        return fig
    else:
        return None