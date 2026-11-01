# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

FastSD CPU: fast Stable Diffusion inference on CPU/Intel GPU/NPU (PyTorch diffusers, OpenVINO, GGUF via stable-diffusion.cpp), with multiple frontends over one engine.

## Setup and running

- Install: `./install.sh` (needs `uv`; creates `./env` with Python 3.11.6, installs CPU torch 2.8.0, then `requirements.txt`). `./install.sh --disable-gui` skips PyQt5 (Termux/headless).
- Activate: `source env/bin/activate`. Always run from the repo root as `python src/app.py ...`.
- Modes (mutually exclusive flags): `-g` Qt GUI, `-w` Gradio web UI (port 7860), `-a` FastAPI server (`--port`, default 8000, docs at `/api/docs`), `-m` MCP server (`/mcp` on `--port`), `-r` realtime UI, `-d` HF demo. No mode flag = one-shot CLI generation; `-i` = interactive CLI.
- CLI examples: `python src/app.py --prompt "a cat" --use_openvino --openvino_lcm_model_id rupeshs/sd-turbo-openvino`; benchmark: `python src/app.py -b --use_openvino`. See `python src/app.py -h`.
- Docker: `docker compose up` (web UI on 7860).
- Env vars: `DEVICE` (`cpu` default; `GPU`/`NPU` → OpenVINO devices; `cuda`/`mps` → torch), `GGUF_THREADS`.

There is no test suite, linter config, or CI. Verify changes by running the relevant mode (CLI one-shot is the fastest smoke test; first run downloads models from Hugging Face).

## Import layout

`src/` is the import root (it is the script dir of `app.py`); modules import each other as top-level names (`import constants`, `from backend.models... import`, `from state import get_settings`). For ad-hoc scripts use `PYTHONPATH=src`. Repo-relative paths (`configs/`, `results/`, `models/`, `lora_models/`, `controlnet_models/`) are resolved from `paths.get_app_path()` (parent of `src/`), not the CWD.

## Architecture

**Request flow:** frontend → `state.get_settings()` / `state.get_context(InterfaceType)` (process-wide singletons in `src/state.py`) → `Context.generate_text_to_image(settings)` (`src/context.py`: timing, settings persistence, safety checker, error capture) → `LCMTextToImage.init()` + `.generate()` in `src/backend/lcm_text_to_image.py`. The API and MCP servers (`src/backend/api/`) each construct their own `Context` at import time.

**Settings:** pydantic models — `models/settings.py:Settings` wraps `backend/models/lcmdiffusion_setting.py:LCMDiffusionSetting` (every generation parameter, including `diffusion_task`, LoRA, ControlNet, GGUF paths). Persisted to `configs/settings.yaml` (gitignored, auto-created from defaults) and re-saved on every generation. New fields need defaults so old YAML files still validate.

**Model lists** come from plain-text files in `configs/` (one HF id or local path per line): `lcm-models.txt`, `stable-diffusion-models.txt` (LCM-LoRA base models), `lcm-lora-models.txt`, `openvino-lcm-models.txt`. GGUF models are discovered from files under `models/gguf/{diffusion,llm,vae}`; the stable-diffusion.cpp shared library (`stable-diffusion.dll` / `libstable-diffusion.so`) is loaded via ctypes from the repo root (`backend/gguf/`).

**Engine modes** are exclusive booleans on `LCMDiffusionSetting`: plain LCM/diffusers (`lcm_model_id`, `backend/pipelines/lcm.py`), `use_lcm_lora` (`backend/pipelines/lcm_lora.py`), `use_openvino`, `use_gguf_model`. `_do_validations` raises if more than one is set (the "delete configs/settings.yaml" error).

**`LCMTextToImage` caches the loaded pipeline** and rebuilds only when one of the tracked `previous_*` fields changes (model ids, TAESD, LoRA, token merging, mode, task type for OpenVINO, …) or `rebuild_pipeline` is set. A new setting that requires a different pipeline must be added to that comparison in `init()`, or it will silently reuse the old pipeline.

**OpenVINO model family is detected by substring of `openvino_lcm_model_id`** (`_is_*_model` helpers): `fibo2` (edit-only), `flux3`, `flux2-klein`, `sana`, `iris` (pixel-space, no VAE), `square` (heterogeneous NPU pipeline `OvHcLatentConsistency`); anything else is SD/SDXL via optimum-intel. Custom pipelines live in `backend/openvino/ov_*_pipeline.py` with loaders in `backend/openvino/pipelines.py`. Adding a new family touches: an `_is_*` predicate, `_do_validations`, the load branch in `init()`, the call branch in `generate()`, a loader in `pipelines.py`, and the model id in `configs/openvino-lcm-models.txt`.

**Tasks** (`DiffusionTask`): text_to_image, image_to_image, edit_image. Image editing is only valid with OpenVINO Flux Klein/Flux3/Fibo2; edit prompt presets come from `configs/edit_image_prompts.json` (`edit_prompts.py`).

**Frontends:** `frontend/webui/` (Gradio; `ui.py` assembles one module per tab; the mode radio flips the engine booleans), `frontend/gui/` (PyQt5), `frontend/cli_interactive.py`. `app.py` deliberately skips heavy imports in realtime mode (`-r`); keep that path light.

`APP_VERSION` and default model ids live in `src/constants.py`.
