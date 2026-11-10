# Explicit compilation artifacts for the CUDA solvers

This CPU-only prototype reuses PTX generated from the CUDA 11.8 translation units used by the warp-observer experiments. It leaves the original solver sources, timing cohorts and driver cache configuration intact.

## Artifact identity

The cache key binds the exact translation unit and its filename, declared entry points, virtual architecture, ordered compiler flags, NVRTC API version, both compiler DLL digests and the source dependency digests. The cache verifies the embedded contract, PTX digest, length and entry points on every load. A controlled timing run must also verify the artifact against the separately frozen build manifest. The embedded digest detects corruption. Provenance comes from the independently frozen manifest.

Three stock modules cover uniform Galerkin, combined Galerkin and FD. The fused Galerkin kernels specialize each schedule for angular lengths 225 and 243. The fused FD kernel uses 243. Together these eight modules cover 24 method, schedule and engine settings. RFM seed, basis dimension, radial quadrature, time step and FD dense or banded solves are runtime inputs within these entries. Different ranks and time schedules still require their own numerical qualification.

The artifact uses a single file. A per-key thread lock and a bounded Windows file lock serialize writes across workers. The writer flushes and syncs a temporary file in the same directory before replacing the cache entry. An absent or corrupt entry can be rebuilt from the same registered inputs in production fallback mode. Strict hit mode fails on a missing, corrupt or mismatched entry. A driver module load failure ends a strict run and keeps its evidence. A later recovery attempt must be recorded separately.

## CPU checks

The synthetic transaction suite exercised concurrent threads and processes, a process exiting while holding the lock, strict corruption rejection, atomic replacement failure, exact repair and input invalidation. Two independent empty-directory CUDA 11.8 builds produced the same eight artifacts byte for byte, totaling 722,351 bytes. The first build spent 0.315–0.347 s compiling each stock module and 0.554–0.870 s per fused module. A fresh CPU process verified all eight artifacts without loading NVRTC, with 0.986–1.130 ms per artifact for cache lookup and content verification. Import, source hashing and process creation sit outside those per-artifact numbers. These observations do not measure fresh GPU solve time or driver module loading.

With the scientific Python interpreter, run these commands from the project root. The output paths must be new for the build and record steps.

~~~powershell
python work/rfm_v8/verify_cuda_ptx_cache.py --output work/rfm_v8/ptx_fixture_replay
python work/rfm_v8/build_cuda_ptx_artifacts.py --build work/rfm_v8/ptx_primary_replay
python work/rfm_v8/build_cuda_ptx_artifacts.py --build work/rfm_v8/ptx_rebuild_replay
python work/rfm_v8/build_cuda_ptx_artifacts.py --verify-hits work/rfm_v8/ptx_primary_replay --record work/rfm_v8/ptx_primary_replay/fresh_cpu_hit.json
python work/rfm_v8/verify_cuda_ptx_rebuild.py work/rfm_v8/ptx_primary_replay work/rfm_v8/ptx_rebuild_replay
~~~

## GPU qualification protocol

The stock and fused paths, RFM and Legendre, and FD dense and banded variants use the same three preparation policies within each matched configuration. Each solve uses a fresh Python process.

| Policy | Preparation charged |
|---|---|
| Cache disabled | Compile the same translation unit from source in each solve |
| Explicit empty cache | Start with a separate empty directory and charge compile, artifact write and verification to the first solve |
| Strict verified hit | Prebuild all registered module keys, record their full cost, and charge lookup and verification in each solve |

Freeze the module sources, compiler, artifact manifest, driver version, precision, schedule, numerical gate and run order before launching the timing cohort. Apply the same observer and warmup policy to each method. Register failed runs and all timing samples. Retain the full interval around process startup and solver return alongside the internal solver clock. Split preparation into context, compile or cache read, driver module load, FFT plan and prime, graph preparation, and other setup where measurable. Record the march and final synchronization. Treat phases within preparation as nested when computing totals. Cache-hit trajectories must reproduce the registered numerical results before a speed comparison.

For a workload of R solves sharing a key, report total cost for the uncached series, the empty-cache first solve plus R−1 verified-hit solves, and prebuild plus R verified-hit solves. Charge a shared prebuild once per distinct key and disclose the exact workload. Reuse across seeds only when keys match. Describe fresh Python process separately from operating-system and driver cache state.

This CUDA 11.8 path emits compute_52 PTX. Loading PTX on the RTX 5060 can still invoke the driver JIT. The PTX artifact avoids repeating NVRTC source compilation. Measure the driver module load separately and retain the default driver cache policy across arms. A separate native-cubin experiment would need a supported target, toolchain provenance, and device qualification. [NVRTC 11.8](https://docs.nvidia.com/cuda/archive/11.8.0/nvrtc/index.html), [NVIDIA Blackwell compatibility guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html), [NVIDIA CUDA environment variables](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html), [Python Windows file locks](https://docs.python.org/3/library/msvcrt.html)

The first T9 attempt reached its 30-second watchdog deadline on the original stock control with caching disabled. The watchdog terminated the owned process and verified its exit. The worker produced no solve return, trajectory or phase marker, so the cause remains unresolved. The frozen summary preserves the attempted slot and all 23 unattempted slots. Neither an explicit empty-cache solve nor a verified-hit solve reached the device, so there is no GPU cache timing or trajectory comparison from this cohort.
