"""Prepare and submit the TUAB handoff for final geometry retry4."""

import prepare_geometry9_1_tuab as prepare
import submit_geometry9_1_tuab_controller as submit


pretrain = prepare.ROOT / "outputs/geometry50_reve3cm_t2_15_20260912_retry4"
campaign = prepare.ROOT / "outputs/geometry9_1_tuab_dropout01_20260912_retry4"
job_id = "27399469"

prepare.PRETRAIN = pretrain
prepare.CAMPAIGN = campaign
prepare.CHECKPOINT = pretrain / "training/checkpoint-epoch-0040.pth"
prepare.PRETRAIN_JOB = job_id

submit.CAMPAIGN = campaign
submit.PRETRAIN = pretrain
submit.PRETRAIN_JOB = job_id
submit.CONTROLLER_JOB_NAME = "geo91-tuab-r4-handoff"


if __name__ == "__main__":
    prepare.prepare()
    submit.submit()
