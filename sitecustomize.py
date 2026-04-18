"""
Workaround for transformers/accelerate nn import issue.
This file is automatically loaded by Python before any other modules.
"""

import sys

# Safely import torch.nn
try:
    import torch
    import torch.nn  # noqa: F401

    # Preemptively inject nn into transformers' namespace
    try:
        import transformers
        if not hasattr(transformers, 'nn'):
            transformers.nn = torch.nn
    except ImportError:
        pass
except ImportError:
    # PyTorch not installed - that's okay for some use cases
    pass
