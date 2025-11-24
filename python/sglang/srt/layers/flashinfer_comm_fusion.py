import logging
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import gpuk

from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.utils import (
    direct_register_custom_op,
    is_flashinfer_available,
    supports_custom_op,
)

logger = logging.getLogger(__name__)

_flashinfer_comm = None
_workspace_manager = None

if is_flashinfer_available():
    try:
        import flashinfer.comm as comm

        _flashinfer_comm = comm
    except ImportError:
        logger.warning(
            "flashinfer.comm is not available, falling back to standard "
            "implementation"
        )


class FlashInferWorkspaceManager:
    def __init__(self):
        self.workspace_tensor = None
        self.ipc_handles = None
        self.world_size = None
        self.rank = None
        self.initialized = False

    def initialize(
        self,
        world_size: int,
        rank: int,
        max_token_num: int,
        hidden_dim: int,
        group=None,
        use_fp32_lamport: bool = False,
    ):
        """Initialize workspace"""
        if self.initialized and self.world_size == world_size:
            return

        if _flashinfer_comm is None:
            logger.warning(
                "FlashInfer comm not available, skipping workspace " "initialization"
            )
            return

        self.cleanup()

        self.ipc_handles, self.workspace_tensor = (
            comm.trtllm_create_ipc_workspace_for_all_reduce_fusion(
                rank,
                world_size,
                max_token_num,
                hidden_dim,
                group=group,
                use_fp32_lamport=use_fp32_lamport,
            )
        )

        self.world_size = world_size
        self.rank = rank
        self.initialized = True

        logger.info(
            f"FlashInfer workspace initialized for rank {rank}, "
            f"world_size {world_size}"
        )

    def cleanup(self):
        """Clean up workspace"""
        if self.initialized and self.ipc_handles is not None:
            try:
                _flashinfer_comm.trtllm_destroy_ipc_workspace_for_all_reduce(
                    self.ipc_handles, group=dist.group.WORLD
                )
            except Exception as e:
                logger.warning(f"Failed to cleanup FlashInfer workspace: {e}")
            finally:
                self.workspace_tensor = None
                self.ipc_handles = None
                self.initialized = False


_workspace_manager = FlashInferWorkspaceManager()


def ensure_workspace_initialized(
    max_token_num: int = 2048, hidden_dim: int = 4096, use_fp32_lamport: bool = False
):
    """Ensure workspace is initialized"""
    if not is_flashinfer_available() or _flashinfer_comm is None:
        return False

    world_size = get_tensor_model_parallel_world_size()
    if world_size <= 1:
        return False

    rank = dist.get_rank()

    if (
        not _workspace_manager.initialized
        or _workspace_manager.world_size != world_size
    ):
        _workspace_manager.initialize(
            world_size=world_size,
            rank=rank,
            max_token_num=max_token_num,
            hidden_dim=hidden_dim,
            use_fp32_lamport=use_fp32_lamport,
        )

    return _workspace_manager.initialized


def flashinfer_allreduce_residual_rmsnorm(
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    max_token_num: int = 2048,
    use_oneshot: Optional[bool] = None,
    trigger_completion_at_end: bool = False,
    fp32_acc: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Use FlashInfer's fused allreduce + residual + RMS norm operation

    Args:
        input_tensor: Input tensor that needs allreduce
        residual: Residual tensor
        weight: RMS norm weight
        eps: RMS norm epsilon
        max_token_num: Maximum token number
        use_oneshot: Whether to use oneshot mode
        trigger_completion_at_end: Whether to trigger completion at end
        fp32_acc: Whether to use fp32 precision

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (norm_output, residual_output)
    """
    if not is_flashinfer_available() or _flashinfer_comm is None:
        logger.debug(
            "FlashInfer not available, falling back to standard " "implementation"
        )
        return None, None

    world_size = get_tensor_model_parallel_world_size()
    if world_size <= 1:
        logger.debug("Single GPU, no need for allreduce fusion")
        return None, None

    assert input_tensor.shape[0] <= max_token_num

    if not ensure_workspace_initialized(
        max_token_num=max_token_num,
        hidden_dim=input_tensor.shape[-1],
        use_fp32_lamport=(input_tensor.dtype == torch.float32),
    ):
        logger.debug("FlashInfer workspace not available")
        return None, None

    token_num, hidden_dim = input_tensor.shape

    residual_out = torch.empty_like(residual)
    norm_out = torch.empty_like(input_tensor)

    _flashinfer_comm.trtllm_allreduce_fusion(
        allreduce_in=input_tensor,
        world_size=world_size,
        world_rank=dist.get_rank(),
        token_num=token_num,
        hidden_dim=hidden_dim,
        workspace_ptrs=_workspace_manager.workspace_tensor,
        launch_with_pdl=True,
        use_oneshot=use_oneshot,
        trigger_completion_at_end=trigger_completion_at_end,
        fp32_acc=fp32_acc,
        pattern_code=(_flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNorm),
        allreduce_out=None,
        residual_in=residual,
        residual_out=residual_out,
        norm_out=norm_out,
        quant_out=None,
        scale_out=None,
        rms_gamma=weight,
        rms_eps=eps,
        scale_factor=None,
        layout_code=None,
    )

    return norm_out, residual_out


def fake_flashinfer_allreduce_residual_rmsnorm(
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    max_token_num: int = 16384,
    use_oneshot: Optional[bool] = None,
    trigger_completion_at_end: bool = False,
    fp32_acc: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    residual_out = torch.empty_like(residual)
    norm_out = torch.empty_like(input_tensor)
    return norm_out, residual_out


if supports_custom_op():
    direct_register_custom_op(
        "flashinfer_allreduce_residual_rmsnorm",
        flashinfer_allreduce_residual_rmsnorm,
        mutates_args=["input_tensor", "residual", "weight"],
        fake_impl=fake_flashinfer_allreduce_residual_rmsnorm,
    )


def cleanup_flashinfer_workspace():
    global _workspace_manager
    if _workspace_manager is not None:
        _workspace_manager.cleanup()

class GPUKManager:
    def __init__(self):
        self.world_size = None
        self.rank = None
        self.dtype = None
        self.initialized = False

    def initialize(
        self,
        world_size: int,
        rank: int,
        dtype: torch.dtype,
    ):
        """Initialize workspace"""
        if self.initialized and self.world_size == world_size:
            return

        self.cleanup()

        self.world_size = world_size
        self.rank = rank
        self.dtype = dtype
        self.dist_env = gpuk.DistributedEnv(rank, world_size, dtype=self.dtype)
        self.initialized = True

    def cleanup(self):
        self.dist_env = None
        self.initialized = False


_gpuk_manager = GPUKManager()


def ensure_gpuk_initialized(dtype):
    """Ensure gpuk is initialized"""
    world_size = get_tensor_model_parallel_world_size()
    if world_size <= 1:
        return False

    rank = dist.get_rank()

    if (
        not _gpuk_manager.initialized
        or _gpuk_manager.world_size != world_size
        or _gpuk_manager.dtype !=  dtype
    ):
        _gpuk_manager.initialize(
            world_size=world_size,
            rank=rank,
            dtype=dtype,
        )

    return _gpuk_manager.initialized


def gpuk_allreduce_residual_rmsnorm_quant(
    allreduce_in: torch.Tensor,
    residual_in: torch.Tensor,
    rms_weight: torch.Tensor,
    eps: float = 1e-6,
    fp8_out: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not ensure_gpuk_initialized(allreduce_in.dtype):
        logger.debug("gpuk not available")
        return None, None, None
    return _gpuk_manager.dist_env.allreduce_add_rms_fused(
        allreduce_in,
        residual_in,
        rms_weight,
        eps,
        fp8_out,
    )


def fake_gpuk_allreduce_residual_rmsnorm_quant(
    allreduce_in: torch.Tensor,
    residual_in: torch.Tensor,
    rms_weight: torch.Tensor,
    eps: float = 1e-6,
    fp8_out: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    residual_out = torch.empty_like(residual_in)
    norm_out = torch.empty_like(allreduce_in)
    scale_out = torch.empty(
                allreduce_in.shape[0],
                1,
                dtype=torch.float32,
                device=allreduce_in.device,
            )
    return residual_out, norm_out, scale_out


direct_register_custom_op(
    "gpuk_allreduce_residual_rmsnorm_quant",
    gpuk_allreduce_residual_rmsnorm_quant,
    mutates_args=["allreduce_in"],
    fake_impl=fake_gpuk_allreduce_residual_rmsnorm_quant,
)
