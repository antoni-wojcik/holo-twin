"""
Study 3: acquire hardware data with the aperture in and the SLM set to 4pi
modulation. This is the main dataset the rest of the automated studies
train and compute against.
"""
from dataclasses import replace

from pipelines.exp.acquire_data import DEFAULT_CONFIG

STUDY = "03_acquire_4pi_aperture"
MODULE = "pipelines.exp.acquire_data"


def build_configs():
    return {
        "run": replace(
            DEFAULT_CONFIG,
            exp_name="ap_4pi",
            exp_description="Aperture in, SLM at 4pi modulation -- main dataset for studies 4 and 6.",
        ),
    }


if __name__ == "__main__":
    from studies.runner import run_study
    run_study(MODULE, STUDY, build_configs())
