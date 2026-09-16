"""
Study 6: module ablation against the study-4 ap_4pi twin, 
repeated across three targets (complex, and checkerboard at
pitch 10 and pitch 120)
"""
from dataclasses import replace

from pipelines.exp.run_cgh import DEFAULT_CONFIG
from src.twin.model import ModuleFlags

STUDY = "06_module_ablation"
MODULE = "pipelines.exp.run_cgh"

# CONFIRM/UPDATE against study 4's actual printed/manifest path once it's run.
TWIN_PATH = r"paper_twin_fit\study_04_4pi_ap_fit\ap_4pi\background_stage\twin_model.pt"

BATCH_NUM = 1
BATCH_SIZE = 4

# (ablation_id, module_flags, overrides)
ABLATIONS = [
    ("lut_off", ModuleFlags(camera=False, background=False, lut=False), {}),
    ("slm_field_off", ModuleFlags(camera=False, background=False, slm_field=False), {}),
    ("pupil_off", ModuleFlags(camera=False, background=False, pupil=False), {}),
    ("pixel_off", ModuleFlags(camera=False, background=False, pixel=False), {}),
    ("background_on", ModuleFlags(camera=False, background=True), {
        "background_aperture_mask_path": r"paper_twin_fit\study_04_4pi_ap_fit\ap_4pi\background_mask\mask.pt",
    }),
    ("all_off", ModuleFlags(camera=False, background=False, lut=False, slm_field=False, pupil=False, pixel=False), {}),
]

# (target_id, pattern, target_path, overrides) -- the three targets each
# ablation above is repeated against.
ABLATION_TARGETS = [
    ("complex", "custom", r"targets\complex.png", {}),
    ("checkerboard_pitch_10", "checkerboard", None, {"checkerboard_pitch": 10}),
    ("checkerboard_pitch_120", "checkerboard", None, {"checkerboard_pitch": 120}),
]


def build_configs(twin_path: str = TWIN_PATH):
    configs = {}
    for ablation_id, flags, ablation_overrides in ABLATIONS:
        for target_id, pattern, target_path, target_overrides in ABLATION_TARGETS:
            run_id = f"{ablation_id}/{target_id}"
            configs[run_id] = replace(
                DEFAULT_CONFIG,
                twin_path=twin_path,
                exp_name=f"study_06_module_ablation/{run_id}",
                pattern=pattern,
                target_path=target_path,
                batch_num=BATCH_NUM,
                batch_size=BATCH_SIZE,
                module_flags=flags,
                **ablation_overrides,
                **target_overrides,
            )
    return configs


if __name__ == "__main__":
    from studies.runner import run_study
    run_study(MODULE, STUDY, build_configs())