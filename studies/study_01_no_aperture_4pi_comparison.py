"""
Study 1 (comparison only -- not used by studies 4/6/7): no aperture, SLM at
4pi modulation. Acquires -> trains -> computes a CGH with the background
module excluded from the display.
"""
import os
from dataclasses import replace

from pipelines.exp.acquire_data import DEFAULT_CONFIG as ACQ_DEFAULT
from pipelines.exp.run_training import DEFAULT_CONFIG as TRAIN_DEFAULT
from pipelines.exp.run_cgh import DEFAULT_CONFIG as CGH_DEFAULT
from src.twin.model import ModuleFlags

from studies.runner import run_isolated, Manifest

STUDY = "01_no_aperture_4pi_comparison"


def main():
    manifest = Manifest()

    acq_cfg = replace(ACQ_DEFAULT, exp_name="no_ap_4pi")
    acquisition_path = run_isolated("pipelines.exp.acquire_data", acq_cfg, "acquire", STUDY, manifest)
    if acquisition_path is None:
        print(f"{STUDY}: acquisition failed -- stopping.")
        return

    train_cfg = replace(TRAIN_DEFAULT, exp_name="no_ap_4pi", acquisition_path=acquisition_path)
    training_path = run_isolated("pipelines.exp.run_training", train_cfg, "train", STUDY, manifest)
    if training_path is None:
        print(f"{STUDY}: training failed -- stopping.")
        return

    twin_path = os.path.join(training_path, "background_stage", "twin_model.pt")
    cgh_cfg = replace(
        CGH_DEFAULT, twin_path=twin_path, exp_name="no_ap_4pi",
        # background excluded from the displayed CGH, per the study spec
        module_flags=ModuleFlags(camera=False, background=False),
    )
    run_isolated("pipelines.exp.run_cgh", cgh_cfg, "cgh", STUDY, manifest)


if __name__ == "__main__":
    main()
