"""Third runtime-only retry using an in-process DataLoader."""

import submit_geometry_reve_retry1 as retry


retry.CAMPAIGN = retry.ROOT / "outputs/geometry50_reve3cm_t2_15_20260912_retry3"
retry.RETRY_OF = "27399239"
retry.JOB_NAME = "eeg-geo50-r3-retry3"
retry.NUM_WORKERS = 0
retry.RUNTIME_FIX = "in-process DataLoader (num_workers=0); avoids AF_UNIX and torch_shm_manager IPC failures"


if __name__ == "__main__":
    retry.submit()
