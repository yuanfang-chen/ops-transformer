import logging
import pytest

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def npu_device_cleanup():
    """Drain NPU error state between tests to prevent cascading failures.

    CANN uses a sticky error model: a device-side kernel failure puts the
    device into an error state that persists until explicitly drained.
    Without this fixture, one test failure would cascade to all remaining tests.
    """
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
