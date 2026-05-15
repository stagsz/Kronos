# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repo Is

Kronos is a decoder-only foundation model for financial K-line (candlestick) data, accepted at AAAI 2026. It uses a two-stage architecture: a VQ-VAE-style tokenizer that quantizes OHLCV data into hierarchical discrete tokens, followed by an autoregressive Transformer that generates forecasts token-by-token.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: for Qlib-based finetuning
pip install pyqlib

# Run prediction example
python examples/prediction_example.py

# Run the web UI
python webui/run.py

# Run all regression tests (downloads model weights from Hugging Face on first run)
pytest tests/test_kronos_regression.py

# Run a single parametrized test
pytest "tests/test_kronos_regression.py::test_kronos_predictor_regression[512]"

# Finetune on CSV data (sequential: tokenizer then predictor)
python finetune_csv/train_sequential.py --config finetune_csv/configs/config_ali09988_candle-5min.yaml

# Finetune on Qlib data (multi-GPU)
torchrun --standalone --nproc_per_node=NUM_GPUS finetune/train_tokenizer.py
torchrun --standalone --nproc_per_node=NUM_GPUS finetune/train_predictor.py

# Qlib backtest
python finetune/qlib_test.py --device cuda:0
```

## Architecture

### Two-Stage Pipeline

**Stage 1 — KronosTokenizer** (`model/kronos.py`, `model/module.py`):
- Encoder: Linear embed → Transformer blocks → `BSQuantizer` (Binary Spherical Quantization)
- BSQ splits each timestep's representation into `s1_bits` (coarse "pre" token) + `s2_bits` (fine "post" token), both stored as integer indices
- Decoder: two parallel paths reconstruct OHLCV from s1-only vs. full (s1+s2) codes

**Stage 2 — Kronos** (`model/kronos.py`):
- `HierarchicalEmbedding`: separate learned embeddings for s1 and s2 vocabularies, fused via a linear projection
- `TemporalEmbedding`: calendar features (minute, hour, weekday, day, month) summed into the sequence
- Stack of `TransformerBlock`: causal self-attention with RoPE + SwiGLU FFN + RMSNorm (pre-norm)
- `DualHead`: projects hidden states to s1 logits; a `DependencyAwareLayer` (cross-attention) conditions s2 prediction on the s1 prediction before projecting to s2 logits
- At inference, `decode_s1` → sample s1 → `decode_s2` → sample s2, one timestep at a time

**KronosPredictor** (`model/kronos.py`):
- High-level wrapper used in all examples and the web UI
- Handles per-series z-score normalization (clip=5), timestamp feature extraction, autoregressive inference loop (`auto_regressive_inference`), and inverse normalization
- `predict()` for single series; `predict_batch()` for parallel batch inference (requires uniform lookback and pred_len)

### Module Inventory (`model/module.py`)
| Class | Role |
|---|---|
| `BinarySphericalQuantizer` | Core VQ with entropy regularisation (BSQ paper) |
| `BSQuantizer` | Wrapper splitting full BSQ into s1/s2 halves |
| `TransformerBlock` | Pre-norm block: RMSNorm → MHA-RoPE → RMSNorm → SwiGLU FFN |
| `HierarchicalEmbedding` | Dual vocab embedding fused for Kronos input |
| `DependencyAwareLayer` | Cross-attention letting s2 attend to transformer context |
| `DualHead` | Separate linear heads for s1 and s2 classification |
| `TemporalEmbedding` | Calendar embedding (fixed sinusoidal or learnable) |

### Model Zoo

| Model | Tokenizer HF ID | Context | Params |
|---|---|---|---|
| Kronos-mini | NeoQuasar/Kronos-Tokenizer-2k | 2048 | 4.1M |
| Kronos-small | NeoQuasar/Kronos-Tokenizer-base | 512 | 24.7M |
| Kronos-base | NeoQuasar/Kronos-Tokenizer-base | 512 | 102.3M |

## Directory Layout

```
model/          Core model: kronos.py (classes), module.py (building blocks)
finetune/       Qlib-based finetuning pipeline; config in finetune/config.py (Config class)
finetune_csv/   CSV-based finetuning; config via YAML + ConfigLoader / CustomFinetuneConfig
examples/       Runnable prediction scripts; examples/yuce/ has historical backtest scripts
webui/          Flask web UI (app.py) with Plotly charts; run via webui/run.py
tests/          Regression tests against pinned HF model revisions
tests/data/     Regression input CSV + expected output CSVs per context length
```

## Key Data Conventions

- Required input columns: `open`, `high`, `low`, `close`. `volume` and `amount` are optional (filled with zeros if absent).
- Timestamps must be a `pd.Series` of `datetime64` values; internally encoded as `[minute, hour, weekday, day, month]`.
- Data is z-score normalised per-series before tokenization and clipped to ±5.
- For `Kronos-small` / `Kronos-base`, the hard context limit is **512 timesteps**. The predictor silently truncates longer inputs.

## Finetuning Paths

**CSV pipeline** (`finetune_csv/`): configure `finetune_csv/configs/*.yaml`, run `train_sequential.py`. All config is read through `CustomFinetuneConfig` → `ConfigLoader`. Supports DDP via `use_ddp: true` in YAML.

**Qlib pipeline** (`finetune/`): edit path/hyperparameter constants in `finetune/config.py`, then run `qlib_data_preprocess.py` → `train_tokenizer.py` → `train_predictor.py` → `qlib_test.py`. Requires a local Qlib data directory.

## Tests

Regression tests in `tests/test_kronos_regression.py` download specific model revisions from Hugging Face (`MODEL_REVISION`, `TOKENIZER_REVISION` constants) and run on CPU. They compare outputs against pre-computed CSVs (`tests/data/regression_output_{ctx}.csv`). To regenerate expected outputs after intentional model changes, run `tests/data/generate_regression_output.py`.
