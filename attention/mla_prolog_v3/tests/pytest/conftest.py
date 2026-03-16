import logging
import pytest

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def npu_device_cleanup():
    """Best-effort NPU cleanup between tests.

    The primary isolation mechanism is subprocess forking in
    _run_npu_isolated() — each NPU invocation runs in a child process
    whose device state dies with it.  This fixture is a secondary safety
    net for any direct NPU usage that bypasses the subprocess wrapper.
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
