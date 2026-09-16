"""
Study 5: CGH target sweep against the study-4 ap_4pi twin (aperture in,
4pi modulation). 
"""
from dataclasses import replace

from pipelines.exp.run_cgh import DEFAULT_CONFIG
from src.twin.model import ModuleFlags

STUDY = "05_target_sweep"
MODULE = "pipelines.exp.run_cgh"

# CONFIRM/UPDATE against study 4's actual printed/manifest path once it's run.
TWIN_PATH = r"paper_twin_fit\study_04_4pi_ap_fit\ap_4pi\background_stage\twin_model.pt"

BATCH_NUM = 5
BATCH_SIZE = 4
BASE_FLAGS = ModuleFlags(camera=False, background=False)

# (run_id, pattern, target_path, overrides)
TARGETS = [
    ("meta", "meta", None, {}),
    ("checkerboard_pitch_10", "checkerboard", None, {"checkerboard_pitch": 10}),
    ("checkerboard_pitch_120", "checkerboard", None, {"checkerboard_pitch": 120}),
    ("complex", "custom", r"targets\complex.png", {}),
    ("usaf", "custom", r"targets\usaf.png", {"dark_penalty": False, "lr_warmup_frac": 0.5, "exposure_time": 0.01}),
    ("grayscale", "custom", r"targets\grayscale.jpg", {"dark_penalty": False}),
]

# Single-hologram variant of tmbgd_full_fov -- see module docstring.
SINGLE_HOLOGRAM_TARGETS = [
    ("complex_single", "custom", r"targets\complex.png", {}),
]


def build_configs(twin_path: str = TWIN_PATH):
    configs = {
        run_id: replace(
            DEFAULT_CONFIG,
            twin_path=twin_path,
            exp_name=f"study_05_target_sweep/{run_id}",
            pattern=pattern,
            target_path=target_path,
            batch_num=BATCH_NUM,
            batch_size=BATCH_SIZE,
            micro_batch_size=DEFAULT_CONFIG.micro_batch_size,
            num_epochs=300,
            module_flags=BASE_FLAGS,
            **overrides,
        )
        for run_id, pattern, target_path, overrides in TARGETS
    }
    configs.update({
        run_id: replace(
            DEFAULT_CONFIG,
            twin_path=twin_path,
            exp_name=f"study_05_target_sweep/{run_id}",
            pattern=pattern,
            target_path=target_path,
            batch_num=1,
            batch_size=1,
            micro_batch_size=1,
            num_epochs=300,
            module_flags=BASE_FLAGS,
            **overrides,
        )
        for run_id, pattern, target_path, overrides in SINGLE_HOLOGRAM_TARGETS
    })
    return configs


if __name__ == "__main__":
    from studies.runner import run_study
    run_study(MODULE, STUDY, build_configs())