from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
YOLO_ROOT = REPO_ROOT / "yolov5"
if str(YOLO_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLO_ROOT))

from models.common import MFGate  # noqa: E402
from utils.datasets import letterbox  # noqa: E402


def load_sample_ids(manifest: Path, count: int, offset: int) -> list[str]:
    sample_ids = [line.strip() for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    return sample_ids[offset: offset + count]


def preprocess_image(path: Path, imgsz: int) -> tuple[np.ndarray, torch.Tensor]:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized, _, _ = letterbox(image_rgb, new_shape=imgsz, auto=False, scaleup=False)
    tensor = torch.from_numpy(np.ascontiguousarray(resized.transpose(2, 0, 1))).float().unsqueeze(0) / 255.0
    return image_rgb, tensor


def save_gate_figure(rgb: np.ndarray, ir: np.ndarray, gate: np.ndarray, save_path: Path) -> None:
    gate_uint8 = np.clip(gate * 255.0, 0, 255).astype(np.uint8)
    gate_color = cv2.applyColorMap(gate_uint8, cv2.COLORMAP_JET)
    gate_color = cv2.cvtColor(gate_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(rgb, 0.6, gate_color, 0.4, 0.0)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), tight_layout=True)
    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    axes[1].imshow(ir, cmap="gray")
    axes[1].set_title("IR")
    axes[2].imshow(gate, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[2].set_title("Gate Map")
    axes[3].imshow(overlay)
    axes[3].set_title("RGB + Gate Overlay")
    for ax in axes:
        ax.axis("off")
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize adaptive modality gate maps for saved checkpoints.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "runs/train/phase2_gate1/weights/best.pt",
        help="Checkpoint path for the trained gate model.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.getenv("VEDAI_DATA_ROOT", REPO_ROOT / "data/phase1_vedai")),
        help="Dataset root containing images/ with *_co.png and *_ir.png files.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "data/manifests/fold02_val.txt",
        help="Manifest file listing sample IDs to visualize.",
    )
    parser.add_argument("--imgsz", type=int, default=512, help="Inference image size.")
    parser.add_argument("--count", type=int, default=8, help="Number of samples to visualize.")
    parser.add_argument("--offset", type=int, default=0, help="Start offset inside the manifest.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs/analysis/phase2_gate1_maps",
        help="Directory for saved visualization panels.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = checkpoint["model"].float().eval()
    gate_module = next((m for m in model.modules() if isinstance(m, MFGate)), None)
    if gate_module is None:
        raise RuntimeError("No MFGate module found in checkpoint model.")

    sample_ids = load_sample_ids(args.manifest, args.count, args.offset)
    device = torch.device("cpu")
    model.to(device)

    for sample_id in sample_ids:
        rgb_path = args.data_root / "images" / f"{sample_id}_co.png"
        ir_path = args.data_root / "images" / f"{sample_id}_ir.png"
        rgb_orig, rgb_tensor = preprocess_image(rgb_path, args.imgsz)
        ir_orig, ir_tensor = preprocess_image(ir_path, args.imgsz)

        with torch.no_grad():
            model(rgb_tensor.to(device), ir_tensor.to(device), "RGB+IR+MF")

        if gate_module.last_gate is None:
            raise RuntimeError("Gate map was not captured during forward pass.")

        gate = gate_module.last_gate.squeeze().cpu().numpy()
        gate = cv2.resize(gate, (rgb_orig.shape[1], rgb_orig.shape[0]), interpolation=cv2.INTER_LINEAR)
        ir_gray = cv2.cvtColor(ir_orig, cv2.COLOR_RGB2GRAY)
        save_gate_figure(rgb_orig, ir_gray, gate, args.output_dir / f"{sample_id}_gate.png")

    print(f"Saved gate visualizations to {args.output_dir}")


if __name__ == "__main__":
    main()
