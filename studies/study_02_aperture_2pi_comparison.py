"""
Study 2: aperture in, SLM at 2pi modulation. Acquires -> trains -> computes 
a CGH, so it can be compared against the aperture'd 4pi run in study 3.
"""
import os
from dataclasses import replace
import numpy as np

from pipelines.exp.acquire_data import DEFAULT_CONFIG as ACQ_DEFAULT
from pipelines.exp.run_training import DEFAULT_CONFIG as TRAIN_DEFAULT
from pipelines.exp.run_cgh import DEFAULT_CONFIG as CGH_DEFAULT

from studies.runner import run_isolated, Manifest

STUDY = "02_aperture_2pi_comparison"


def main():
    manifest = Manifest()

    acq_cfg = replace(ACQ_DEFAULT, exp_name="ap_2pi")
    acquisition_path = run_isolated("pipelines.exp.acquire_data", acq_cfg, "acquire", STUDY, manifest)
    if acquisition_path is None:
        print(f"{STUDY}: acquisition failed -- stopping.")
        return

    train_cfg = replace(
        TRAIN_DEFAULT, exp_name="ap_2pi", acquisition_path=acquisition_path,
        init_lut_scale=2 * np.pi,  # see the CHECK BEFORE RUNNING note above
    )
    training_path = run_isolated("pipelines.exp.run_training", train_cfg, "train", STUDY, manifest)
    if training_path is None:
        print(f"{STUDY}: training failed -- stopping.")
        return

    twin_path = os.path.join(training_path, "background_stage", "twin_model.pt")
    cgh_cfg = replace(CGH_DEFAULT, twin_path=twin_path, exp_name="ap_2pi")
    run_isolated("pipelines.exp.run_cgh", cgh_cfg, "cgh", STUDY, manifest)


if __name__ == "__main__":
    main()
