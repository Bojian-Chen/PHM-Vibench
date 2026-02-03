"""Utilities for instantiating models from configuration."""

from __future__ import annotations

import importlib
import os
from typing import Any

import torch
from ..utils.utils import get_num_classes
from ..utils.registry import Registry




def resolve_model_module(args_model: Any) -> str:
    """Return the Python import path for the model module."""
    return f"src.model_factory.{args_model.type}.{args_model.name}"


def model_factory(args_model: Any, metadata: Any):
    """Instantiate a model by name.

    Parameters
    ----------
    args_model : Namespace
        Configuration namespace with at least ``name`` and ``type``
        fields. Other attributes are passed to the model's ``Model``
        constructor.
    metadata : Any
        Dataset metadata, used here only to compute ``num_classes``.

    Returns
    -------
    nn.Module
        Instantiated model ready for training.
    """
    args_model.num_classes = get_num_classes(metadata)
    # key = f"{args_model.type}.{args_model.name}"


    module_path = resolve_model_module(args_model)
    model_module = importlib.import_module(module_path)
    model_cls = model_module.Model

    try:
        model = model_cls(args_model, metadata)
        
        if hasattr(args_model, "weights_path") and args_model.weights_path:
            weights_path = args_model.weights_path
            if os.path.exists(weights_path):
                try:
                    load_ckpt(model, weights_path)
                except Exception as e:  # pragma: no cover - runtime safeguard
                    print(f"加载权重时出错: {e}")
        
        return model
    
    except Exception as e:
        raise RuntimeError(f"创建模型实例时出错: {str(e)}")
    

def load_ckpt(model, ckpt_path):
    """Load weights from ``ckpt_path`` into ``model``.

    Parameters
    ----------
    model : nn.Module
        Model instance to be updated.
    ckpt_path : str
        Path to a PyTorch checkpoint file.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file {ckpt_path} does not exist.")
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    # Handle Lightning checkpoints and raw state_dicts.
    state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    if not isinstance(state_dict, dict):
        raise RuntimeError("Unsupported checkpoint format: expected a state_dict-like mapping.")

    # Strip common prefixes from Lightning/DDP checkpoints.
    if any(k.startswith("network.") for k in state_dict):
        state_dict = {k.replace("network.", "", 1): v for k, v in state_dict.items() if k.startswith("network.")}
    if any(k.startswith("model.") for k in state_dict):
        state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items() if k.startswith("model.")}
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items() if k.startswith("module.")}

    model_dict = model.state_dict()
    matched_dict = {}
    skipped = []
    for name, param in state_dict.items():
        if name in model_dict:
            matched_dict[name] = param
        else:
            skipped.append((name, "not in model"))
    # 加载匹配的权重
    model.load_state_dict(matched_dict, strict=False)
    # 打印跳过的参数
    if skipped:
        print("跳过以下不匹配的参数：")
        for name, model_sz in skipped:
            print(f"  {name}: checkpoint vs model {model_sz}")
    print(f"已加载匹配的权重: {ckpt_path}")
