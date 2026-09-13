"""Final runtime-only retry with short file-descriptor IPC sockets."""

import submit_geometry_reve_retry1 as retry


retry.CAMPAIGN = retry.ROOT / "outputs/geometry50_reve3cm_t2_15_20260912_retry4"
retry.RETRY_OF = "27399319"
retry.JOB_NAME = "eeg-geo50-r3-retry4"
retry.NUM_WORKERS = 8
retry.RUNTIME_FIX = "short cached multiprocessing tempdir with file_descriptor sharing; configured 8 workers restored"


if __name__ == "__main__":
    retry.submit()
