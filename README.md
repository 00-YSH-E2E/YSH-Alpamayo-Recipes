<!-- 코드 설명 (2026-11-08 추가) -->
## 이 저장소는

NVIDIA Alpamayo Recipes(Alpamayo 1·1.5 SFT·RL·양자화)의 사본에, PAD 형식으로 바꾼 nuScenes CoC 데이터로 Alpamayo 1.5 를 LoRA SFT 하는 설정을 더한 것입니다. 2026-11-01 부터는 `YSH-Alpamayo1.5-FineTuning/recipes/` 에서 이어갑니다.

| 항목 | 내용 |
|---|---|
| 목적 | upstream 레시피로 A1.5 SFT·양자화를 돌리고, nuScenes CoC LoRA SFT 를 시험 |
| 핵심 코드 | `recipes/alpamayo1_5_sft/train_hf.py` HF Trainer SFT (+LoRA·resume) · `recipes/alpamayo1_5_sft/configs/sft_stage1_coc.yaml` nuScenes CoC LoRA 설정 · `src/alpamayo/data/pai.py` `PAIDatasetAllEvents` (clip 의 모든 event 사용) · `recipes/alpamayo1_5_quant/` FP8·NVFP4 양자화 (upstream) · `scripts/` PAI 다운로드·체크포인트 변환 |
| 실행 | `torchrun --nproc_per_node 8 -m alpamayo1_5_sft.train_hf --config-path pkg://alpamayo1_5_sft/configs --config-name sft_stage1_nav` (README) — 우리 run 은 `--config-name sft_stage1_coc` |
| 입력 → 출력 | A1.5 체크포인트 + `/workspace/Data/nusc_pad` (`reasoning/ood_reasoning_qc.parquet`) → `output_stage1_coc_lora/` checkpoint, `adapter_final/` LoRA |
| 환경 | 레시피마다 uv venv (`uv venv a1_5_sft` → `uv sync --active`) · HF Trainer + DeepSpeed (CoC LoRA 는 ZeRO 없이 97 GB GPU 1장) |
| 원본 | `NVlabs/alpamayo-recipes` 를 07-18 통째 복사 (upstream 히스토리 없음) · 08-25 `train_hf.py` LoRA·resume, `PAIDatasetAllEvents`, `sft_stage1_coc.yaml`, hydra run 로그 추가 |
| 상태 | 마지막 코드 변경 2026-08-25 · `main` 만 · 보관 (이후 변경은 FineTuning `recipes/`) |

---

# Alpamayo Recipes

A collection of end-to-end Alpamayo recipes for multiple versions (v1, v1.5, and beyond), designed
to help developers quickly build, adapt, and deploy Alpamayo-based applications. This repo
brings together battle-tested workflows across the Alpamayo ecosystem, including post-training
recipes (supervised fine-tuning and reinforcement learning), quantization recipes, etc.
Whether you are experimenting locally or building a full production stack, this repository is
intended as the primary starting point for developers to learn, customize, and extend
Alpamayo for their own use cases.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the repository layout, recipe packaging conventions,
and guidance on adding new recipes for released Alpamayo models.

## Recipes

Each recipe folder contains its own README with installation and training instructions.

| Recipe                                                              | Description                                                                             |
| ------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| [`recipes/alpamayo1_sft/`](recipes/alpamayo1_sft/README.md)         | Alpamayo 1 supervised fine-tuning (HuggingFace Trainer + DeepSpeed)                     |
| [`recipes/alpamayo1_5_sft/`](recipes/alpamayo1_5_sft/README.md)     | Alpamayo 1.5 SFT (HuggingFace Trainer + DeepSpeed)                                      |
| [`recipes/alpamayo1_x_rl/`](recipes/alpamayo1_x_rl/README.md)       | Alpamayo 1 and 1.5 RL post-training (Cosmos-RL / GRPO)                                  |
| [`recipes/alpamayo1_5_quant/`](recipes/alpamayo1_5_quant/README.md) | Alpamayo 1.5 quantization (Model Optimizer Toolkit / FP8 / NVFP4 + FP8 Mixed Precision) |

## Utility Scripts

| Script                                          | Purpose                                              |
| ----------------------------------------------- | ---------------------------------------------------- |
| `scripts/curate_pai_samples.py`                 | Curate a subset of PAI samples                       |
| `scripts/convert_checkpoint.py`                 | Convert between Alpamayo 1 and 1.5 checkpoints       |
| `scripts/convert_release_config_to_training.py` | Convert a release checkpoint to training format      |
| `scripts/convert_cosmos_rl_checkpoint.py`       | Convert a Cosmos-RL checkpoint to HuggingFace format |
| `scripts/download_pai.py`                       | Download the Physical AI AV dataset from HuggingFace |
# YSH-Alpamayo-Recipes
