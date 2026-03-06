"""Compatibility shim for optional Composer dependency.

This allows inference-only code paths to run when MosaicML Composer is not
fully importable in mixed dependency environments.
"""

from __future__ import annotations

import torch.nn as nn

try:
    from composer.models import ComposerModel as ComposerModel  # type: ignore
except Exception:
    class _NoOpLogger:
        def log_metrics(self, *_args, **_kwargs) -> None:
            return

    class ComposerModel(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            self.logger = _NoOpLogger()
