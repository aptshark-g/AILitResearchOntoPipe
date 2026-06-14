"""Core engine components — no AI dependency.

These modules form the backbone of Literature Cortex:
- config: Configuration loading from file/env
- state: Pipeline state machine with checkpoint/resume
- engine: Main workflow orchestration (A→G phases)
"""

from lcortex.core.config import Config, LLMConfig, WorkspaceConfig, get_config
from lcortex.core.state import PipelineState, PhaseStatus, save_state, load_state, resume_from_checkpoint
from lcortex.core.engine import run_pipeline

__all__ = [
    "Config",
    "LLMConfig",
    "WorkspaceConfig",
    "get_config",
    "PipelineState",
    "PhaseStatus",
    "save_state",
    "load_state",
    "resume_from_checkpoint",
    "run_pipeline",
]
