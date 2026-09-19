# Experiment execution and result reporting

When the user asks to run an experiment, treat monitoring and result publication as part of the requested workflow.

- Monitor without reserving a GPU. Use a local CPU/background monitor or a CPU-only dependent job.
- Every downstream run starts immediately at its configured base learning rate. Do not add warmup logic, warmup configuration keys, or warmup-labelled downstream filenames. Pretraining schedules are outside this rule.
- Do not publish a final aggregate until every requested dataset has exactly five completed seeds and every seed has a readable `result.json`.
- Write results below `outputs/results/`.
- Use one folder per dataset named `MMDDHHmm-dataset-pretrain_alias`, for example `09151628-tuab-knn37-warmup0`.
- Each dataset folder must contain `results.csv` and `results.md`. Update the global `outputs/results/RESULTS.md` in the same operation.
- The comparison table starts with `Metric`, `우리 (<pretrain alias>)`, followed by prior-study model columns such as `CBraMod`, `CSBrain`, and `REVE`.
- In every results file, comparison table, and aggregate index, list datasets in this exact order: CHB-MIT, SIENA, PHYSIONET-MI, TUEV, TUAB, FACED, SEED-V, Mental Arithmetic, ISRUC, HMC. Omit datasets absent from a campaign without changing the relative order; place any other datasets after these. Apply this to CSV rows as well as Markdown sections. Corresponding slugs are `chb`, `siena`, `physionet_mi` (or `physio`), `tuev`, `tuab`, `faced`, `seedv` (or `seed-v`), `mentalarithmetic` (or `stress`), `isruc`, `hmc`.
- Rank methods independently for every metric. In Markdown, render the highest mean in **bold** and the second-highest mean with `<u>underline</u>`. CSV stores plain numeric means, population SDs, sample counts, and explicit rank columns; formatting must never be the only rank encoding.
- Select checkpoints using the campaign's validation selector. Never select a checkpoint, hyperparameter, or method from test performance.
- Preserve the five individual seed values in `seed_results.csv` for auditability.
- If a job finishes without all expected artifacts, record the missing dataset/seeds and do not label the experiment complete.
- The four GR2 campaigns `gr2-mjde-d4-geometry`, `gr2-d2-static`, `gr2-d2-patch-scalar`, and `gr2-d2-patch-dimension` are main-result candidates. Their downstream five-seed campaigns use A100.
- For later ablation or comparison campaigns, pretraining uses `a100_short,a100_long`; downstream uses L40S. Route `tuab`, `chb`, and `tuev` to `gl40s_long`. Route `tusl`, `tusz`, `seedv`, `faced`, `mentalarithmetic`, `physionet_mi`, `isruc`, `hmc`, and `siena` to `gl40s_dev,gl40s_short`.
- Keep per-dataset L40S downstream runtime measurements. Until five-seed observations exist, label any duration used for scheduling as an estimate rather than an empirical average.
