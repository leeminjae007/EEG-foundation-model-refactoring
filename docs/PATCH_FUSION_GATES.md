# MJDE patch-dependent fusion gates

Full MJDE keeps its three stages and twelve attention/FFN blocks. At each stage,
the two branch outputs A = S→T and B = T→S are fused as `g*A + (1-g)*B`.
The default remains `static_feature`. Two explicit configurations implement
input-conditioned route mixing:

| `encoder.fusion_gate` | Gate shape per stage | Definition | Gate parameters, D=200, 3 stages | Total model parameters |
| --- | --- | --- | ---: | ---: |
| `static_feature` (default) | `[1,1,1,D]` | `sigmoid(theta[stage])` | 600 | 6,612,830 |
| `patch_scalar` | `[B,C,T,1]` | `sigmoid(Linear(2D,1)(concat(A,B)))` | 1,203 | 6,613,433 |
| `patch_feature` | `[B,C,T,D]` | `sigmoid(Linear(2D,D)(concat(A,B)))` | 240,600 | 6,852,830 |

Each stage has its own projection. Within a stage, the same projection weights
are shared across all samples, electrodes and time patches. The coefficients
are computed from the branch outputs, rather than learned separately for each
position. Each scalar mixes all D features together; each feature gate can mix
each feature separately. Both paths continue to include spatial and temporal
processing. The coefficients describe route mixing, not a patch's causal
importance or a pure spatial-versus-temporal importance score.

## Initialization and masking

All gate projection weights and biases start at zero, so every coefficient is
initially 0.5. Nonzero gradients train the projections, after which coefficients
depend on the input. This is an intentional neutral initialization, not a frozen
gate. Explicit zero Parameters prevent the existing model-wide Kaiming
initializer from replacing this initialization and consume no extra random
numbers. Tokenizer, SHPE, attention/FFN and decoder initial tensors and RNG state
are therefore exactly equal to the static baseline when initialized with the
same seed. The initial outputs and shared-parameter gradients are also equal.

The existing context-only masks apply to both paths and the fused result. Hidden
tokens and absent channels cannot enter attention through the gate. Statistics
include visible tokens only, including when a whole temporal/spatial row or an
entire sample is hidden. No extra normalization, dropout, pooling, auxiliary loss,
or attention changes are introduced by these gate variants.

## Configurations and checkpoints

- `configs/pretrain_gr2_patch_scalar.yaml`
- `configs/pretrain_gr2_patch_feature.yaml`
- Reference: `configs/pretrain_gr2_geometry.yaml`

The two standalone native-pretraining configurations change only
`encoder.fusion_gate` and the output folder relative to the reference. All keep
D=200, SHPE 100+100, geometry radius 35–75 degrees, 2–10 seconds, 50% masking,
independent rank RNG, 40 epochs, batch 128/GPU, accumulation 1, LR 5e-4 to 1e-5,
weight decay 0.05 and dropout 0.1. The intended four-GPU global batch is 512.

`src/encoder.py` selects the gate implementation. The default static state-dict
keys remain unchanged. Native dynamic pretrain checkpoints load strictly into
the existing downstream builder, which reconstructs the encoder from the saved
pretrain config. Optimizer and RNG checkpoint roundtrips preserve the next step
exactly. Different gate modes have incompatible strict state dictionaries and
must not silently resume each other's runs. In the ablation adapter, dynamic
gates currently require full MJDE; other encoder arms reject this combination
rather than silently discarding the gate configuration.

## Observability and verification

The regular diagnostic cadence logs gate mean, standard deviation, range,
saturation fraction, variation across patches, variation across features, and
projection weight/bias gradient RMS for every stage. It does not retain full
gate tensors or autograd graphs between steps.

`tests/test_patch_fusion.py` checks both output granularities, input dependence,
weight sharing across positions, exact initial equivalence, hidden-token
isolation, empty attention rows, gradient flow, checkpoint/optimizer/RNG resume,
strict downstream loading, and rejection of unsupported configurations.

`scripts/validate_patch_gates.py` additionally trains each variant for one real
TUEG CPU optimizer step, verifies strict loading and all gate gradients, and
measures the learned gates on real data. Its checkpoints are explicitly marked
`partial_epoch_smoke` and are never used as completed pretraining or resume
weights. Results are implementation validation, not evidence of downstream
performance superiority. See `outputs/results/audits/patch-fusion-20260917/`.
