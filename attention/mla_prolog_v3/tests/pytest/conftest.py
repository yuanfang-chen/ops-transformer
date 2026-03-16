import logging
import pytest

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def npu_device_cleanup():
    """Drain NPU error state between tests to prevent cascading failures."""
    yield
    try:
        import torch
        torch.npu.synchronize()
    except Exception as e:
        logger.warning("npu_device_cleanup: synchronize raised %s: %s",
                       type(e).__name__, e)
    try:
        import torch
        torch.npu.empty_cache()
    except Exception as e:
        logger.warning("npu_device_cleanup: empty_cache raised %s: %s",
                       type(e).__name__, e)
