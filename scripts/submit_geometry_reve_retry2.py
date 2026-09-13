"""Second runtime-only retry after the file-descriptor sharing failure."""

import submit_geometry_reve_retry1 as retry


retry.CAMPAIGN = retry.ROOT / "outputs/geometry50_reve3cm_t2_15_20260912_retry2"
retry.RETRY_OF = "27399149"
retry.JOB_NAME = "eeg-geo50-r3-retry2"


if __name__ == "__main__":
    retry.submit()
