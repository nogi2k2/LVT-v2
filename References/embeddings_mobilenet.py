"""Embedding utilities using MobileNetV3-Small when available.

This module exposes a small API that tries to use torchvision/torch to run a
pretrained MobileNetV3-Small on CPU to produce feature vectors for image
patches. If torch/torchvision are not available, a fast OpenCV-based fallback
produces a simple resized-and-flattened vector.

Public API:
 - load_embedder() -> (model, preprocess) or (None, fallback_fn)
 - embed_crop(bgr_crop, model) -> np.ndarray
 - embed_boxes(original_bgr, boxes, model) -> list[np.ndarray]
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple
import numpy as np
import cv2


def _make_fallback():
    """Return a fallback embedder that resizes to 64x64 RGB, converts to float,
    normalizes to [0,1] and flattens to a vector.
    """
    def embed_fallback(bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        if h == 0 or w == 0:
            return np.zeros((64 * 64 * 3,), dtype=np.float32)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (64, 64), interpolation=cv2.INTER_AREA)
        vec = small.astype(np.float32) / 255.0
        return vec.ravel()

    return embed_fallback


def load_embedder() -> Tuple[Optional[object], Callable[[np.ndarray], np.ndarray]]:
    """Try to load a MobileNetV3-Small embedder (torch). If not available,
    return (None, fallback_fn).

    The returned callable takes a BGR numpy array crop and returns a 1-D numpy
    float32 vector.
    """
    try:
        import torch
        import torchvision.transforms as T
        try:
            # torchvision may expose mobilenet_v3_small
            from torchvision.models import mobilenet_v3_small
        except Exception:
            # Older/newer versions may differ; fallback to models API
            from torchvision import models
            mobilenet_v3_small = models.mobilenet_v3_small

        # ImageNet normalization
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        preprocess = T.Compose([
            T.ToPILImage(),
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

        # Load pretrained model and remove classifier to get feature vector
        model = mobilenet_v3_small(pretrained=True)
        # Replace classifier with identity so forward returns features
        try:
            model.classifier = torch.nn.Identity()
        except Exception:
            # If structure differs, try to find last feature extractor
            pass
        model.eval()

        def embed_torch(bgr: np.ndarray) -> np.ndarray:
            # Convert BGR to RGB
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            inp = preprocess(rgb)
            inp = inp.unsqueeze(0)  # batch dim
            with torch.no_grad():
                out = model(inp)
            # out may be a tensor; convert to numpy
            vec = out.cpu().numpy().ravel().astype(np.float32)
            return vec

        return model, embed_torch
    except Exception:
        return None, _make_fallback()


def embed_crop(bgr_crop: np.ndarray, model_or_none) -> np.ndarray:
    """Embed a single BGR crop. model_or_none is either a torch model (returned
    by load_embedder) or None; in the latter case a fallback is used.
    """
    # If model_or_none is a tuple (model, fn) caller may pass fn; handle either
    if callable(model_or_none):
        return model_or_none(bgr_crop)
    # Otherwise load an embedder dynamically
    model, fn = load_embedder() if model_or_none is None else (model_or_none, None)
    if fn is None:
        # model provided, build fn using model
        # create a closure similar to load_embedder
        import torch
        import torchvision.transforms as T
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        preprocess = T.Compose([
            T.ToPILImage(),
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

        def fn_local(bgr: np.ndarray) -> np.ndarray:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            inp = preprocess(rgb)
            inp = inp.unsqueeze(0)
            with torch.no_grad():
                out = model(inp)
            return out.cpu().numpy().ravel().astype(np.float32)

        return fn_local(bgr_crop)
    else:
        return fn(bgr_crop)


def embed_boxes(original_bgr: np.ndarray, boxes: List[Tuple[int, int, int, int]], model_or_none=None) -> List[np.ndarray]:
    """Compute embeddings for each bounding box in `boxes` from the original
    BGR image. Returns a list of 1-D numpy float32 arrays.
    """
    embeddings: List[np.ndarray] = []
    for (x, y, w, h) in boxes:
        if w <= 0 or h <= 0:
            embeddings.append(np.zeros((1,), dtype=np.float32))
            continue
        crop = original_bgr[y:y+h, x:x+w]
        vec = embed_crop(crop, model_or_none)
        embeddings.append(vec)
    return embeddings
