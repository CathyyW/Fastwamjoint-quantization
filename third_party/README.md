# Pinned native dependencies

Run `python scripts/setup_dependencies.py`, or `bash bash/setup_kernels.sh` to fetch
and compile. Revisions are pinned in `versions.lock.json`, copied from the existing
working kernel dependencies. No precompiled `.so`, dependency `.git` directories,
or full dependency copies are committed to this repository.

- CUTLASS: NVIDIA/cutlass; headers used by both kernels.
- Fast Hadamard Transform: Dao-AILab/fast-hadamard-transform; W4A4 uses its C++
  headers. This static fused path does not require building its separate Python
  extension. Each fetched repository retains its own license.

The exact had12/had28 constants in `rht_constants.py` and `rht_constants.cuh`
originate from the upstream RHT utilities (`fake_quant/hadamard_utils.py`), commit
`5008669b08c1f11f9b64d52d16fddd47ca754c5a` (Apache-2.0). The corresponding license
is retained under `licenses/`. RHT names in this project refer to the local
randomized Hadamard transform, not to a port of the full comparison algorithm.

For a wheel install outside this source tree, set `CUTLASS_PATH` to the checkout
root and `FAST_HADAMARD_PATH` to the fast-hadamard-transform checkout root. Both
loaders accept those environment variables. CUDA sources themselves are included
in the wheel. `CUDA_HOME` must point to a CUDA Toolkit with nvcc.
