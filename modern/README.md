# MSDDN — release-compatible modernization

Pinned reference: [original public commit 834bfb9](https://github.com/alexhe101/MSDDN/tree/834bfb9bc2e1a4c3b30efb8891c1e5b5326e6c19). This is an implementation refactor, not a paper-inspired redesign.

## Compatibility contract

- Isolate spatial coupling, LU mixing, Fourier fusion and reconstruction.
- Preserve original spatial/frequency guidance order, FFT normalization, rfft2/irfft2, amplitude/phase operations, epsilon placement and final magnitude.
- Keep all registered parameters, including inactive tensors. LU masks/identity become non-persistent buffers so they follow module.to() without changing checkpoint keys or performing repeated device transfers.
- Preserve the released CALayer implementation, including its pooled-residual behavior; do not replace it with a different attention block.

The state dict names and shapes remain the same as the released `Net`. The strict loader handles bare weights or the original `net` wrapper and an all-keys DataParallel prefix. Unused registered tensors are deliberately retained. The repository does not contain public trained checkpoints. Tests use the actual released model's initialized weights, not recovered pretrained weights.

### Installation

Python 3.10+ and PyTorch 2.1+. Install a PyTorch build appropriate for your CPU or GPU. SciPy is required for MSDDN's original LU initialization and for original-reference tests.

```bash
python -m pip install -e './modern[dev]'
python -m pytest modern/tests -v -rs
```

## Input and output contract

Use one NPZ per co-registered scene, with finite floating CHW arrays in the same band order and radiometric range as the trained model. No implicit normalization, clipping, resize, quantization or image-mode conversion.

| Key | Shape | Required |
|:--|:--|:--|
| ms | [4,h,w] | Always |
| pan | [1,4h,4w] | Always |
| target | [4,4h,4w] | Supervised training |

For the released 4-band weights, the expected bands are RGB + near-infrared; preserve the exact original dataset channel order and normalization. Use FP32 for reference comparisons. Use CPU or CUDA, matching the model and input devices. At least two LR spatial pixels are required by the smallest instance-normalized feature map. Same-sized scenes must be batched together. The output NPZ has an unbatched floating-point `hrms` array with all four bands.

```python
import torch
from msddn import build_model, predict, load_weights

model = build_model(device="cpu")
load_weights(model, "your_checkpoint.pth")
model.eval()
ms = torch.rand(1,4,32,32,device="cpu")
pan = torch.rand(1,1,128,128,device="cpu")
with torch.inference_mode():
    hrms = predict(model, ms, pan)
```

Random inputs above only demonstrate the tensor contract. They are not satellite results.

```bash
msddn infer --device cpu --checkpoint your_checkpoint.pth --input scene.npz --output result.npz
msddn benchmark --device cpu --height 128 --image-width 128 --batch 1
msddn train --device cpu --data paired_train --output runs/train
```

The new optional NPZ trainer keeps the active released default objective: L1 with sum reduction divided by 2× the configured batch size, Adam at 5e-4, gradient clipping 40, and StepLR(200, 0.1), stepped once per epoch. It is single-device and accepts prepaired data; original data degradation, augmentation, validation metrics, optimizer-resume behavior and exact training run are not reimplemented. The original solver/config are preserved for historical experiments. The modern `--checkpoint` argument initializes weights, not optimizer resume. Benchmarks synchronize CUDA and report synthetic latency and allocated memory, not claimed speedups. No AMP, compiler, alternate kernel or tiling is enabled silently.

## Validation

The suite compares the refactored model against the actual published source at width 4 and width 32: state-dict keys, strict loading, outputs, input gradients and every active parameter gradient. FFT and coupling computations are compared without replacing their operators. It also checks the CLI and one-step synthetic training. These are regression tests on synthetic inputs; they are not remeasurements of the paper's dataset quality.

[Automated test run](https://github.com/alexhe101/MSDDN/actions/runs/45547411741). Results must be interpreted according to the specific commit and test environment.

## Original code and publication

- [Original model](https://github.com/alexhe101/MSDDN/blob/834bfb9bc2e1a4c3b30efb8891c1e5b5326e6c19/model/msddn.py)
- [Original configuration](https://github.com/alexhe101/MSDDN/blob/834bfb9bc2e1a4c3b30efb8891c1e5b5326e6c19/option.yml)
- [Publication](https://doi.org/10.1109/TGRS.2023.3273334)
