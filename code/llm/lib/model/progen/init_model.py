import sys
import os
from pathlib import Path

import torch

from lib.model.progen.path import PROGEN_PATH

MODEL_PATH = PROGEN_PATH + "ProGen/progen/progen2/checkpoints/progen2-small"


def init_model(device: torch.device):
    parent_dir = os.path.dirname(Path(__file__))
    sys.path.append(parent_dir + "/" + PROGEN_PATH)
    from ProGen.progen.progen2.models.progen.modeling_progen import ProGenForCausalLM

    path = parent_dir + "/" + MODEL_PATH

    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ProGenForCausalLM.from_pretrained(
        path,
        revision="float16",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)

    return model
