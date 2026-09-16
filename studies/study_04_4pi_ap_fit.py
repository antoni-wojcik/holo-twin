"""
Study 4: fit the digital twin against the (unchanged) hardware acquisitions
from studies 1 and 3, using the current model.py. 
"""
from dataclasses import replace

from pipelines.exp.run_training import DEFAULT_CONFIG

STUDY = "04_4pi_ap_fit"
MODULE = "pipelines.exp.run_training"

ACQUISITION_PATHS = {
    "ap_4pi": r"paper_training_data\ap_4pi",
    "no_ap_4pi": r"paper_training_data\no_ap_4pi",
}

NUM_SAMPLES = 150


def build_configs(acquisition_paths: dict = ACQUISITION_PATHS):
    return {
        dataset: replace(
            DEFAULT_CONFIG,
            exp_name=f"study_04_4pi_ap_fit/{dataset}",
            acquisition_path=acquisition_paths[dataset],
            num_noise_samples=NUM_SAMPLES,
            iterations=(300, 100),
            use_crosstalk_fast_approx=False,
        )
        for dataset in acquisition_paths
    }


if __name__ == "__main__":
    from studies.runner import run_study
    run_study(MODULE, STUDY, build_configs())