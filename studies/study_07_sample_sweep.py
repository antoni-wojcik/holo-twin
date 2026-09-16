"""
Study 7: training sample-count sweep, ap_4pi only. Uses the same
acquisition as study 4's ap_4pi run (unaffected by model.py changes, not
re-acquired), swept over SAMPLE_COUNTS. 
"""
from dataclasses import replace

from pipelines.exp.run_training import DEFAULT_CONFIG

STUDY = "07_sample_sweep"
MODULE = "pipelines.exp.run_training"

# Same acquisition as study 4's ap_4pi entry -- CONFIRM/UPDATE together with it.
ACQUISITION_PATH = r"paper_training_data\ap_4pi"

SAMPLE_COUNTS = [4, 8, 10, 20, 50, 100]

# Extra entry appended after the sweep: same 20-sample point, but with the
# SLM field coarsened by SLM_FIELD_DIVISOR. Isolates how much of the
# small-sample behaviour is the field's degrees of freedom rather than the
# number of training pairs.
SLM_FIELD_DIVISOR_SAMPLES = 20
SLM_FIELD_DIVISOR = 40


def build_configs(acquisition_path: str = ACQUISITION_PATH):
    common = dict(
        acquisition_path=acquisition_path,
        iterations=(300, 100),
        use_crosstalk_fast_approx=False,
        run_stages=(True, True, False),  # skip the background stage in the comparison
    )

    configs = {
        str(n): replace(
            DEFAULT_CONFIG,
            exp_name=f"study_07_sample_sweep/{n}_samples",
            num_noise_samples=n,
            **common,
        )
        for n in SAMPLE_COUNTS
    }

    configs[f"{SLM_FIELD_DIVISOR_SAMPLES}_field_div{SLM_FIELD_DIVISOR}"] = replace(
        DEFAULT_CONFIG,
        exp_name=(
            f"study_07_sample_sweep/{SLM_FIELD_DIVISOR_SAMPLES}_samples"
            f"_field_div{SLM_FIELD_DIVISOR}"
        ),
        num_noise_samples=SLM_FIELD_DIVISOR_SAMPLES,
        slm_field_divisor=SLM_FIELD_DIVISOR,
        **common,
    )

    return configs


if __name__ == "__main__":
    from studies.runner import run_study
    run_study(MODULE, STUDY, build_configs())