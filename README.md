# COS529 Advanced Computer Vision - Vehicle Detection in Aerial Imagery

Multimodal aerial vehicle detection project built around RGB-IR VEDAI experiments and a custom YOLO-based fusion model.

## Project Overview

This project studies vehicle detection in aerial imagery using both **color (RGB)** and **infrared (IR)** inputs. The work started from a custom YOLOv5-based multimodal detector and evolved into a reproducible experimental pipeline with modality ablations and targeted fusion improvements.

Current focus areas:
- Multimodal RGB-IR vehicle detection on VEDAI
- Small-object detection in aerial imagery
- Feature-level fusion and adaptive modality gating
- Reproducible training and evaluation from WSL Ubuntu

## Additional Docs

- [SETUP.md](/C:/Users/nithi/OneDrive/Documents/cvprinceton/COS529_PROJECT/SETUP.md)
- [USAGE.md](/C:/Users/nithi/OneDrive/Documents/cvprinceton/COS529_PROJECT/USAGE.md)
- [RESULTS.md](/C:/Users/nithi/OneDrive/Documents/cvprinceton/COS529_PROJECT/RESULTS.md)
- [README_COS529.md](/C:/Users/nithi/OneDrive/Documents/cvprinceton/COS529_PROJECT/yolov5/README_COS529.md)

## Phase 1

Phase 1 is the reproducibility cleanup pass. The goal is to make the multimodal baseline runnable from a clean WSL Ubuntu environment before changing the research method.

### What was standardized

- Portable VEDAI manifest generation in [prepare_vedai_manifests.py](/C:/Users/nithi/OneDrive/Documents/cvprinceton/COS529_PROJECT/scripts/prepare_vedai_manifests.py)
- Sanitized single-class Phase 1 dataset generation in [prepare_phase1_dataset.py](/C:/Users/nithi/OneDrive/Documents/cvprinceton/COS529_PROJECT/scripts/prepare_phase1_dataset.py)
- WSL setup script in [setup_wsl.sh](/C:/Users/nithi/OneDrive/Documents/cvprinceton/COS529_PROJECT/scripts/setup_wsl.sh)
- Canonical Phase 1 train script in [run_phase1_train.sh](/C:/Users/nithi/OneDrive/Documents/cvprinceton/COS529_PROJECT/scripts/run_phase1_train.sh)
- Canonical dataset YAML in [vedai_phase1.yaml](/C:/Users/nithi/OneDrive/Documents/cvprinceton/COS529_PROJECT/yolov5/data/vedai_phase1.yaml)
- Dataset loader path resolution no longer depends on old Colab absolute paths

### WSL quick start

From the repo root:

```bash
bash scripts/setup_wsl.sh
source .venv/bin/activate
bash scripts/run_phase1_train.sh
```

The run script automatically prefers the full dataset at `../MultiModalFusion/data/VEDAI` if the local `data/VEDAI` folder is incomplete. You can also override the dataset root explicitly:

```bash
export VEDAI_DATA_ROOT=/absolute/path/to/VEDAI
bash scripts/run_phase1_train.sh
```

### Current dataset caveat

The checked-in `data/VEDAI` folder is only a partial copy. The complete 512x512 image-label set is available in `../MultiModalFusion/data/VEDAI`, and the Phase 1 run script is set up to use it automatically when present.

## Phase 1 Modality Benchmark

These are the canonical single-class `vehicle` results on the sanitized Phase 1 VEDAI setup.

| Mode | Run folder | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| --- | --- | ---: | ---: | ---: | ---: |
| RGB+IR+MF | `runs/train/phase1_baseline1` | 0.8328 | 0.7872 | 0.8696 | 0.5087 |
| RGB-only | `runs/train/phase1_rgb2` | 0.8198 | 0.7677 | 0.8564 | 0.4991 |
| IR-only | `runs/train/phase1_ir` | 0.8235 | 0.7018 | 0.8099 | 0.4642 |

Takeaway: the multimodal `RGB+IR+MF` model is stronger than both single-modality ablations on recall and mAP.

### Re-running the modality experiments

```bash
source .venv/bin/activate
bash scripts/run_phase1_mf.sh
bash scripts/run_phase1_rgb.sh
bash scripts/run_phase1_ir.sh
```

## Phase 2 Architecture Study

Phase 2 compares three targeted upgrades on top of the multimodal baseline:
- `FEM`: FFCA-inspired feature enhancement on the final P3 small-object branch
- `Gate`: adaptive modality gating at the RGB/IR fusion stage
- `FEM + Gate`: both changes together

| Model | Run folder | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| --- | --- | ---: | ---: | ---: | ---: |
| Phase 1 baseline | `runs/train/phase1_baseline1` | 0.8328 | 0.7872 | 0.8696 | 0.5087 |
| Phase 2 + FEM | `runs/train/phase2_fem` | 0.8379 | 0.7946 | 0.8766 | 0.5150 |
| Phase 2 + Gate | `runs/train/phase2_gate1` | 0.8585 | 0.7866 | 0.8792 | 0.5181 |
| Phase 2 + FEM + Gate | `runs/train/phase2_fem_gate` | 0.8408 | 0.7756 | 0.8701 | 0.5126 |

Takeaway: both single upgrades improve on the Phase 1 multimodal baseline, and the simple adaptive gating variant is currently the strongest result. Combining `FEM` and `Gate` did not outperform `Gate` alone in this setup.

### Running the Phase 2 variants

```bash
source .venv/bin/activate
bash scripts/run_phase2_fem.sh
bash scripts/run_phase2_gate.sh
bash scripts/run_phase2_fem_gate.sh
```

## Gate Visualization

The gate-analysis utility exports RGB / IR / gate-map / overlay panels from the saved adaptive-gating checkpoint.

Script:
- [visualize_gate_maps.py](/C:/Users/nithi/OneDrive/Documents/cvprinceton/COS529_PROJECT/scripts/visualize_gate_maps.py)

Example command:

```bash
source .venv/bin/activate
python scripts/visualize_gate_maps.py --count 8
```

Default output directory:
- `runs/analysis/phase2_gate1_maps`

## Current Best Model

The strongest current result is:
- [phase2_gate1](/C:/Users/nithi/OneDrive/Documents/cvprinceton/COS529_PROJECT/runs/train/phase2_gate1)

Metrics:
- Precision: `0.8585`
- Recall: `0.7866`
- mAP@0.5: `0.8792`
- mAP@0.5:0.95: `0.5181`

## License

This project is licensed under the MIT License. See [LICENSE](/C:/Users/nithi/OneDrive/Documents/cvprinceton/COS529_PROJECT/LICENSE).
