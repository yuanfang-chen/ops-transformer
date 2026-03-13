import sys
from pathlib import Path

import pytest
import torch


PYTEST_DIR = Path(__file__).resolve().parent
if str(PYTEST_DIR) not in sys.path:
    sys.path.insert(0, str(PYTEST_DIR))

import aclnnMlaPrologV3Ref
import prologv3_generalized


def test_pa_blk_bsnd_bsnd_accepts_matrix_and_flat_indices():
    cache = torch.zeros((4, 4, 1, 8), dtype=torch.float32)
    input_ = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    index_matrix = torch.tensor([[2], [1]], dtype=torch.int64)
    index_flat = index_matrix.reshape(-1)

    generalized_matrix = prologv3_generalized.scatter_pa_blk_bsnd(
        cache.clone(), input_, index_matrix, 4, 2
    )
    generalized_flat = prologv3_generalized.scatter_pa_blk_bsnd(
        cache.clone(), input_, index_flat, 4, 2
    )
    old_matrix = aclnnMlaPrologV3Ref.scatter_pa_blk_bsnd(
        cache.clone(), input_, index_matrix, 4, 2
    )
    old_flat = aclnnMlaPrologV3Ref.scatter_pa_blk_bsnd(
        cache.clone(), input_, index_flat, 4, 2
    )

    assert torch.equal(generalized_matrix, generalized_flat)
    assert torch.equal(old_matrix, old_flat)
    assert torch.equal(generalized_matrix, old_matrix)


def test_pa_blk_nz_tnd_accepts_flattened_page_indices():
    cache = torch.zeros((4, 4, 1, 8), dtype=torch.float32)
    input_ = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    actual_seq_len = torch.tensor([3, 8], dtype=torch.int32)
    flat_page_index = torch.tensor([0, 3, 1], dtype=torch.int64)

    generalized = prologv3_generalized.scatter_pa_blk_nz(
        cache.clone(), input_, flat_page_index, actual_seq_len, 2, 16
    )
    old = aclnnMlaPrologV3Ref.scatter_pa_blk_nz(
        cache.clone(), input_, flat_page_index, actual_seq_len, 2, 16
    )

    assert torch.equal(generalized, old)


def test_pa_blk_nz_tnd_rejects_matrix_page_indices():
    cache = torch.zeros((4, 4, 1, 8), dtype=torch.float32)
    input_ = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    actual_seq_len = torch.tensor([3, 8], dtype=torch.int32)
    matrix_page_index = torch.tensor([[0, 3], [1, 2]], dtype=torch.int64)

    with pytest.raises(AssertionError):
        prologv3_generalized.scatter_pa_blk_nz(
            cache.clone(), input_, matrix_page_index, actual_seq_len, 2, 16
        )

    with pytest.raises(AssertionError):
        aclnnMlaPrologV3Ref.scatter_pa_blk_nz(
            cache.clone(), input_, matrix_page_index, actual_seq_len, 2, 16
        )
