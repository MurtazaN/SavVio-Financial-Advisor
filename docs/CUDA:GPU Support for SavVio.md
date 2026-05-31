# CUDA/GPU Support for SavVio ML & data Pipeline

## Context
The model pipeline trains XGBoost, LightGBM, and XGB-Linear models. Currently everything runs CPU-only. The Dockerfile and docker-compose.yml already have commented-out GPU scaffolding. The goal is to evaluate multiple approaches for enabling CUDA when a GPU is available, while keeping the pipeline functional on CPU-only machines.

---

## Option 1: Runtime Auto-Detection (Recommended)

**How it works:** Add a utility function that checks for GPU availability at runtime (`nvidia-smi` or library-level queries). Inject `tree_method="gpu_hist"` / `device="cuda"` into model params only when a GPU is detected. No user configuration needed.

### Files to modify
| File | Change |
|------|--------|
| [config.py](model_pipeline/src/config.py) | Add `USE_GPU = "auto"` setting (auto / force-cpu / force-gpu) |
| [train.py](model_pipeline/src/core_models/train.py) | Add `_detect_gpu()` helper; inject `tree_method`/`device` into model constructors; add to `VALID_PARAMS` |
| [optuna_tuner.py](model_pipeline/src/core_models/optuna_tuner.py) | Same GPU params injected into each objective's `XGBClassifier`/`LGBMClassifier` |
| [Dockerfile](model_pipeline/Dockerfile) | No change to default; keep GPU option commented |
| [docker-compose.yml](model_pipeline/docker-compose.yml) | No change to default; keep GPU deploy section commented |

### Detection logic (in `train.py`)
```python
import shutil, subprocess

def _detect_gpu() -> bool:
    """Check if a CUDA GPU is available at runtime."""
    if Config.USE_GPU == "force-cpu":
        return False
    if Config.USE_GPU == "force-gpu":
        return True
    # auto: probe nvidia-smi
    if not shutil.which("nvidia-smi"):
        return False
    try:
        subprocess.run(["nvidia-smi"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
```

### How GPU params get injected
- **XGBoost tree:** `tree_method="gpu_hist"` (falls back to `"hist"` on CPU)
- **XGBoost linear:** No GPU acceleration available for `gblinear` booster -- skip
- **LightGBM:** `device="gpu"` (falls back to `"cpu"`)

### Pros
- Zero config for users -- works on CPU and GPU machines without changes
- Single Docker image works everywhere (GPU params are just constructor args, not compile-time)
- XGBoost pip package already includes CUDA support since v2.0 -- no rebuild needed
- Easy to override via env var (`USE_GPU=force-cpu`)

### Cons
- LightGBM pip wheel does NOT include GPU kernels -- `device="gpu"` will error unless LightGBM is compiled with CUDA (see Dockerfile change)
- `nvidia-smi` probe adds ~200ms startup latency
- Auto-detection can give false positives (driver installed but no free VRAM)

### Impact when GPU is NOT available
- `_detect_gpu()` returns `False`, all models use CPU params -- **zero behavioral change** from today
- No extra dependencies needed, no import errors
- XGBoost `tree_method="hist"` is already the default on CPU

---

## Option 2: Environment Variable Toggle (Simple)

**How it works:** A single env var `GPU_ENABLED=true` in `.env` / docker-compose controls GPU mode. No auto-detection.

### Files to modify
| File | Change |
|------|--------|
| [config.py](model_pipeline/src/config.py) | `GPU_ENABLED = os.getenv("GPU_ENABLED", "false").lower() == "true"` |
| [train.py](model_pipeline/src/core_models/train.py) | If `Config.GPU_ENABLED`, add GPU params to constructors |
| [optuna_tuner.py](model_pipeline/src/core_models/optuna_tuner.py) | Same conditional GPU param injection |
| [docker-compose.yml](model_pipeline/docker-compose.yml) | Add `GPU_ENABLED: ${GPU_ENABLED:-false}` to ml-trainer env |
| `.env` | Add `GPU_ENABLED=false` |

### Pros
- Simplest implementation -- one boolean check, no detection logic
- Explicit -- user knows exactly what mode they're in
- No risk of false positives

### Cons
- User must manually set the flag -- forgetting = always CPU even with a GPU
- Mismatch risk: `GPU_ENABLED=true` on a CPU-only host will crash at model.fit()
- Requires separate `.env` files or compose overrides for CPU vs GPU deployments

### Impact when GPU is NOT available
- If `GPU_ENABLED=false` (default): **zero behavioral change**
- If `GPU_ENABLED=true` on CPU-only host: **XGBoost will silently fall back to CPU** (`gpu_hist` degrades gracefully in xgboost >=2.0), but **LightGBM will crash** with `device="gpu"` if not CUDA-compiled

---

## Option 3: Dual Dockerfile with Compose Profiles

**How it works:** Two separate Dockerfiles (`Dockerfile` for CPU, `Dockerfile.gpu` for GPU). Use docker compose profiles to select at launch time: `docker compose --profile gpu up`.

### Files to modify/create
| File | Change |
|------|--------|
| `Dockerfile.gpu` (new) | Based on `nvidia/cuda:12.1.1-devel-ubuntu22.04`, installs CUDA-compiled LightGBM |
| [docker-compose.yml](model_pipeline/docker-compose.yml) | Add `ml-trainer-gpu` service with `profiles: [gpu]` and the NVIDIA deploy block |
| [config.py](model_pipeline/src/config.py) | Add `USE_GPU` env var |
| [train.py](model_pipeline/src/core_models/train.py) | Conditional GPU params (same as Option 1/2) |
| [optuna_tuner.py](model_pipeline/src/core_models/optuna_tuner.py) | Conditional GPU params |
| [model-requirements.txt](model_pipeline/model-requirements.txt) | Keep as-is (CPU). GPU Dockerfile installs GPU LightGBM separately |

### Pros
- Clean separation -- CPU image stays slim (~800MB), GPU image is larger (~4GB) but has everything
- LightGBM GPU actually works (compiled from source with CUDA in GPU image)
- Compose profiles make switching easy: `--profile gpu` vs default
- CI/CD can build and test both images

### Cons
- Two Dockerfiles to maintain -- drift risk
- Longer GPU image build time (~10-15 min for LightGBM CUDA compile)
- More complex compose file
- Still need code-level changes to pass GPU params to models

### Impact when GPU is NOT available
- Users run default profile (no `--profile gpu`) -- **CPU Dockerfile, zero change from today**
- If someone accidentally runs `--profile gpu` without NVIDIA runtime, `docker compose up` will fail at container start (clear error from Docker, not a silent bug)

---

## Option 4: Multi-Stage Dockerfile with Build Arg

**How it works:** Single Dockerfile with a `BUILD_TARGET` arg. `docker compose build --build-arg BUILD_TARGET=gpu` selects the CUDA base and compiles GPU LightGBM. Default builds CPU.

### Dockerfile sketch
```dockerfile
ARG BUILD_TARGET=cpu

FROM python:3.11-slim AS base-cpu
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS base-gpu
RUN apt-get update && apt-get install -y python3 python3-pip python3-dev build-essential cmake ...

FROM base-${BUILD_TARGET} AS final
WORKDIR /app
COPY model-requirements.txt .
RUN pip install -r model-requirements.txt
# GPU-only: recompile LightGBM with CUDA
ARG BUILD_TARGET=cpu
RUN if [ "$BUILD_TARGET" = "gpu" ]; then \
      pip install lightgbm --no-binary lightgbm \
        --config-settings=cmake.define.USE_CUDA=ON; \
    fi
```

### Files to modify
| File | Change |
|------|--------|
| [Dockerfile](model_pipeline/Dockerfile) | Rewrite as multi-stage |
| [docker-compose.yml](model_pipeline/docker-compose.yml) | Add `args: BUILD_TARGET: ${BUILD_TARGET:-cpu}` |
| [config.py](model_pipeline/src/config.py), [train.py](model_pipeline/src/core_models/train.py), [optuna_tuner.py](model_pipeline/src/core_models/optuna_tuner.py) | Same GPU param injection as other options |

### Pros
- Single Dockerfile -- no drift risk
- Build arg makes intent explicit at image build time
- CPU image stays lean by default

### Cons
- Multi-stage Docker builds are harder to debug
- `BUILD_TARGET` is a build-time decision, not runtime -- must rebuild to switch
- LightGBM CUDA compilation still takes ~10-15 min
- More complex Dockerfile to understand

### Impact when GPU is NOT available
- Default `BUILD_TARGET=cpu` -- **zero change from today**
- If built with `gpu` but run without NVIDIA runtime, container fails to start (same as Option 3)

---

## Comparison Matrix

| Criteria | Option 1: Auto-Detect | Option 2: Env Var | Option 3: Dual Dockerfile | Option 4: Multi-Stage |
|----------|----------------------|-------------------|--------------------------|----------------------|
| **Code changes** | Medium | Small | Medium | Medium |
| **Dockerfile changes** | None (for XGBoost) | None | New file | Rewrite |
| **Works without config** | Yes | No (must set var) | No (must pick profile) | No (must set build arg) |
| **LightGBM GPU** | Only if CUDA-compiled | Only if CUDA-compiled | Yes (compiled in GPU image) | Yes (compiled in GPU image) |
| **XGBoost GPU** | Yes (pip wheel has CUDA) | Yes | Yes | Yes |
| **CPU-only impact** | None | None | None | None |
| **Risk of crash on CPU** | Low (auto-detects) | Medium (misconfigured env) | Low (Docker fails early) | Low (Docker fails early) |
| **Maintenance burden** | Low | Low | Medium (2 Dockerfiles) | Medium (complex Dockerfile) |
| **Image size (CPU)** | ~800MB (unchanged) | ~800MB (unchanged) | ~800MB (unchanged) | ~800MB (unchanged) |
| **Image size (GPU)** | N/A (same image) | N/A (same image) | ~4GB | ~4GB |

---

## Recommendation

**Option 1 (Auto-Detect) + Option 3 (Dual Dockerfile) combined** gives the best experience:

1. **Code layer (Option 1):** Auto-detect GPU at runtime in `train.py` and `optuna_tuner.py` -- XGBoost GPU works immediately since the pip wheel includes CUDA kernels.
2. **Docker layer (Option 3):** Add `Dockerfile.gpu` + compose profile for when LightGBM GPU is also needed -- this handles the compile-time requirement.
3. **Override (from Option 2):** Expose `USE_GPU` env var for manual control (`auto`/`force-cpu`/`force-gpu`).

This means:
- **CPU-only machines:** Change nothing. Pipeline works exactly as today.
- **GPU machines, XGBoost only:** Just enable NVIDIA runtime in compose. Auto-detection kicks in. No rebuild.
- **GPU machines, XGBoost + LightGBM:** Use `--profile gpu` to get the CUDA-compiled LightGBM image.

---

## Data Pipeline: GPU for Sentence-Transformers (vector_embed.py)

### Current State
- Base image: `apache/airflow:3.1.7` (CPU-only)
- `sentence-transformers` is installed, which pulls PyTorch **CPU-only** wheel by default
- Model: `all-MiniLM-L6-v2` (22MB, 384-dim) -- lightweight
- Batch size: 64, chunked in groups of 5,000
- Runs as an Airflow `PythonOperator` task (final DAG step)
- Worker memory limit: 4GB

### Key difference from model_pipeline
PyTorch/sentence-transformers has **built-in CUDA auto-detection**: `SentenceTransformer.encode()` automatically uses GPU if `torch.cuda.is_available()` returns `True`. **No code changes to vector_embed.py are needed** -- the only requirement is getting a CUDA-enabled PyTorch into the container.

---

### Data Pipeline GPU Option A: Install CUDA PyTorch Wheel (Recommended)

**How it works:** Replace the default CPU PyTorch (pulled by sentence-transformers) with the CUDA wheel in `data-requirements.txt`.

#### Files to modify
| File | Change |
|------|--------|
| [data-requirements.txt](data_pipeline/data-requirements.txt) | Add `--extra-index-url https://download.pytorch.org/whl/cu121` and pin `torch` with CUDA |
| [Dockerfile](data_pipeline/Dockerfile) | Switch base to a CUDA-capable image (Airflow doesn't ship CUDA) |
| [docker-compose.yaml](data_pipeline/docker-compose.yaml) | Add NVIDIA deploy block to the `airflow-worker` service |

#### Pros
- **Zero code changes** to `vector_embed.py` -- PyTorch auto-detects GPU
- Sentence-transformers `.encode()` moves tensors to GPU transparently
- Embedding generation for ~100K+ reviews goes from minutes to seconds
- Batch size can be increased (e.g., 512) for better GPU utilization

#### Cons
- **Image size balloons** ~2-3GB for CUDA PyTorch wheel alone
- Airflow base image (`apache/airflow`) is Debian-based, not CUDA-native -- need a custom image or multi-stage build
- Only benefits the embedding task; all other Airflow DAG tasks are I/O-bound (DB loads, API calls) and don't benefit from GPU
- **Overkill for small datasets** -- `all-MiniLM-L6-v2` encodes ~10K sentences/sec on CPU already

#### Impact when GPU is NOT available
- If CUDA PyTorch is installed but no GPU present: **PyTorch silently falls back to CPU** -- `torch.cuda.is_available()` returns `False`, `.encode()` runs on CPU exactly as today
- The only cost is the larger Docker image (~2-3GB extra for unused CUDA libraries)

---

### Data Pipeline GPU Option B: Separate GPU Worker Service

**How it works:** Add a dedicated `airflow-worker-gpu` service in docker-compose with a GPU-enabled image. Route only the embedding task to this worker using Airflow queue routing.

#### Files to modify
| File | Change |
|------|--------|
| [docker-compose.yaml](data_pipeline/docker-compose.yaml) | Add `airflow-worker-gpu` service with NVIDIA deploy, pointing to a GPU Dockerfile |
| `Dockerfile.gpu` (new in data_pipeline) | CUDA base + Airflow + PyTorch CUDA + sentence-transformers |
| [data_pipeline_airflow.py](data_pipeline/dags/data_pipeline_airflow.py) | Set `queue="gpu"` on the `generate_load_embeddings` task |
| [vector_embed.py](data_pipeline/dags/src/database/vector_embed.py) | No changes needed |

#### Pros
- CPU workers stay lightweight -- only the embedding worker has the GPU image
- Clean separation of concerns -- GPU cost is isolated
- Airflow queue routing is a built-in feature, no hacks needed
- Can scale GPU workers independently

#### Cons
- More complex Airflow/Compose setup
- Two different worker images to maintain
- Queue routing adds operational complexity
- For a small academic project, this is over-engineered

#### Impact when GPU is NOT available
- If the GPU worker service is not started (no `--profile gpu`): the embedding task sits in the `gpu` queue with no consumer -- **it will hang indefinitely** unless a fallback queue is configured
- Needs a default queue fallback or conditional queue assignment

---

### Data Pipeline GPU Option C: Do Nothing (Keep CPU)

**How it works:** Leave the data pipeline as-is. CPU is adequate for the current workload.

#### Justification
- `all-MiniLM-L6-v2` is a 22MB model -- not compute-heavy
- At batch_size=64, CPU throughput is ~10K sentences/sec
- For ~50K-100K product/review embeddings, total time is ~10-30 seconds on CPU
- The embedding task runs once per pipeline execution, not continuously
- The real bottleneck in the DAG is I/O (API fetches, DB writes), not embedding compute

#### Pros
- Zero changes, zero risk
- Keeps Airflow image small and fast to build
- No GPU infrastructure needed for data pipeline
- Embedding latency is already negligible in the overall DAG runtime

#### Cons
- If dataset grows to millions of rows, CPU embedding becomes a bottleneck
- Larger models (e.g., `all-mpnet-base-v2` at 420MB, or future LLM-based embeddings) would benefit significantly from GPU

#### Impact when GPU is NOT available
- **No impact** -- this IS the current behavior

---

### Data Pipeline Recommendation

**Option C (Do Nothing)** for now -- the embedding workload is too small to justify GPU infrastructure in the data pipeline. The `all-MiniLM-L6-v2` model processes the entire dataset in seconds on CPU.

**Revisit with Option A** when:
- Dataset grows beyond ~500K rows
- You switch to a larger embedding model (e.g., `all-mpnet-base-v2`, `e5-large`)
- Embedding generation becomes a measurable DAG bottleneck

---

## Verification
1. **CPU path:** Run `docker compose up ml-trainer` (default) -- confirm training completes with `tree_method=hist` / `device=cpu` in MLflow logged params
2. **GPU path (XGBoost):** On a GPU host, uncomment NVIDIA deploy block, run pipeline -- confirm `tree_method=gpu_hist` appears in MLflow
3. **GPU path (LightGBM):** Build with `--profile gpu`, confirm `device=gpu` in MLflow
4. **Force-CPU override:** Set `USE_GPU=force-cpu`, confirm CPU params even on GPU host
5. **Unit tests:** Mock `_detect_gpu()` to test both branches without needing real hardware
