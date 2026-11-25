"""Compare the refactor with the exact public release, not a hand-transcribed model."""
import hashlib
import importlib
from pathlib import Path
import sys
import types

import pytest
import torch
from msddn import build_model, predict, load_weights

ROOT = Path(__file__).resolve().parents[2]
RELEASE_BLOBS = {
    "model/msddn.py": "b4890c37e003450983384d3b2ac0c0e7b2aa895f",
    "model/refine.py": "bc18e91f814419c13c0ccbae0205b72ac064578d",
    "model/modules.py": "9e9ec6520212c8800b60bb2b2f0ff98530982d13",
    "model/thops.py": "7f83218c6a5ad513947bd4f2f8d63fa6f564f5aa"
}

def git_blob_sha(path):
    b = path.read_bytes()
    return hashlib.sha1(f"blob {len(b)}\0".encode() + b).hexdigest()


def test_published_sources_unchanged():
    for path, sha in RELEASE_BLOBS.items():
        assert git_blob_sha(ROOT/path) == sha


def original_net(height=None, width=None):
    package_name = "_published_msddn_reference"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / "model")]
    sys.modules[package_name] = package
    for name in ("thops", "modules", "refine", "msddn"):
        module = types.ModuleType(package_name + "." + name)
        module.__package__ = package_name
        sys.modules[module.__name__] = module
        p = Path(package.__path__[0])/(name+".py")
        source = p.read_text(encoding="utf-8")
        # Remove only unused optional visualization dependencies.
        source = "\n".join(line for line in source.splitlines()
                         if not line.startswith(("import cv2", "import torchvision")))
        exec(compile(source, str(p), "exec"), module.__dict__)
    return sys.modules[package_name+".msddn"].Net

@pytest.mark.parametrize("width", [4, 32])
def test_release_forward_backward(width):
    torch.manual_seed(12)
    old = original_net()(num_channels=4, base_filter=width, args=None).eval()
    new = build_model(width=width).eval()
    assert list(old.state_dict()) == list(new.state_dict())
    new.load_state_dict(old.state_dict(), strict=True)
    a = torch.rand(2, 4, 4, 5, requires_grad=True)
    b = torch.rand(2, 1, 16, 20, requires_grad=True)
    aa = a.detach().clone().requires_grad_()
    bb = b.detach().clone().requires_grad_()
    x = old(a, None, b)
    y = predict(new, aa, bb)
    torch.testing.assert_close(x,y,atol=1e-6,rtol=1e-6)
    x.square().mean().backward()
    y.square().mean().backward()
    torch.testing.assert_close(a.grad,aa.grad,atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(b.grad,bb.grad,atol=1e-6,rtol=1e-5)
    for (name,p),(new_name,q) in zip(old.named_parameters(),new.named_parameters()):
        assert name == new_name
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad,q.grad,atol=1e-6,rtol=1e-5,msg=name)
    print("max absolute error:", (x-y).abs().max().item())

def test_native_inference_and_training(tmp_path):
    import numpy as np
    from msddn.cli import main
    inp = tmp_path/"scene.npz"
    np.savez(inp, ms=np.ones((4,4,4),np.float32)*.5,
             pan=np.ones((1,16,16),np.float32)*.5,
             target=np.ones((4,16,16),np.float32)*.5)
    ckpt = tmp_path/"model.pth"
    torch.save(build_model(width=4).state_dict(), ckpt)
    out = tmp_path/"result.npz"
    main(["infer","--width","4","--checkpoint",str(ckpt),"--input",str(inp),"--output",str(out)])
    with np.load(out) as data:
        assert data["hrms"].shape == (4,16,16)
        assert np.isfinite(data["hrms"]).all()
    out.unlink()
    main(["train","--width","4","--data",str(tmp_path),"--output",str(tmp_path/"run"),
          "--epochs","1","--batch","1"])
    assert (tmp_path/"run"/"latest.pth").is_file()
