import unittest
from pathlib import Path

import numpy as np
import torch
from torch.testing import assert_close

from continualUtils.benchmarks.datasets.preprocess import preprocess_input


class TestNormalizeDataset(unittest.TestCase):
    def test_normalize_np_image(self):
        # Create a 3D tensor
        np_img = np.array(
            [
                [[255.0, 255.0, 255.0], [255.0, 255.0, 255.0]],
                [[255.0, 255.0, 255.0], [255.0, 255.0, 255.0]],
            ]
        )

        # Call the normalization function
        normalized_img = preprocess_input(np_img)

        # Define expectation
        expected_img = np.array(
            [
                [[2.249, 2.429, 2.640], [2.249, 2.429, 2.640]],
                [[2.249, 2.429, 2.640], [2.249, 2.429, 2.640]],
            ]
        )

        # Assert that the output is as expected
        assert_close(normalized_img, expected_img, rtol=1e-03, atol=1e-03)


if __name__ == "__main__":
    unittest.main()
