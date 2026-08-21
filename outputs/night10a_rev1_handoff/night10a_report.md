# Night-10A REV1 report

Terminal status: `IMPLEMENTATION_SEMANTICS_INVALID`.

## Plain-language result

No ARI, NMI or Q result was opened. The label firewall remained closed, so there is no valid claim that A1, tonsil, D1 or P22 improved or declined.

The registered unified adapter worked dimensionally on the three RNA+protein datasets, producing 63 locked CUDA trainings and 42 completed fixed-endpoint transforms before the systemic issue was recognized. On P22, every one of the 21 registered trainable cells failed at its first forward: the G04 private views have 128 columns while the authoritative R02 reference embedding has 64. The adapter concatenated 128+128+64=320 columns but its frozen layer expected 384 (Q06: 336 actual versus 400 expected).

This also reveals a retrospective P0-REV1 coverage defect: P0 checked the real P22 endpoint and a synthetic CUDA adapter round-trip separately, but never ran the adapter on the real mixed-dimensional P22 inputs. Therefore the earlier P0 PASS is retained as history but marked a false pass.

## Why execution stopped

Adding a projection for the 64-dimensional reference or changing input dimensions would modify the trainable architecture after formal scientific outputs had already begun. The taskbook prohibits that. The run therefore stopped with 0 label reads, 0 Stage M, 0 R2, 0 scientific retries and 0 fallback. Existing successful, failed and partial artifacts are preserved and SHA-indexed; none is used as scientific evidence.

## Required future repair

A new authority revision must specify a single cross-family dimension contract, for example a preregistered fixed/non-trainable reference projection or a family-independent adapter that accepts an explicitly registered reference dimension. It must add real A1 and real P22 forward/loss/checkpoint tests before any formal training. This cannot be repaired inside Night-10A REV1.
