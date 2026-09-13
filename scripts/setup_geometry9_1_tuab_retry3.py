"""Prepare and submit the TUAB handoff for successful geometry retry3."""

import prepare_geometry9_1_tuab as prepare
import submit_geometry9_1_tuab_controller as submit


pretrain = prepare.ROOT / "outputs/geometry50_reve3cm_t2_15_20260912_retry3"
campaign = prepare.ROOT / "outputs/geometry9_1_tuab_dropout01_20260912_retry3"
job_id = "27399319"

prepare.PRETRAIN = pretrain
prepare.CAMPAIGN = campaign
prepare.CHECKPOINT = pretrain / "training/checkpoint-epoch-0040.pth"
prepare.PRETRAIN_JOB = job_id

submit.CAMPAIGN = campaign
submit.PRETRAIN = pretrain
submit.PRETRAIN_JOB = job_id
submit.CONTROLLER_JOB_NAME = "geo91-tuab-r3-handoff"


if __name__ == "__main__":
    prepare.prepare()
    submit.submit()
