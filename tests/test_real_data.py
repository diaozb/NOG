import tempfile
import unittest
from pathlib import Path

import torch

from src.distributed.real_data import CappedL1SVM, load_libsvm_dense


class RealDataTests(unittest.TestCase):
    def test_parser_and_row_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.libsvm"
            path.write_text("1 1:3 3:4\n-1 2:2\n", encoding="utf-8")
            features, labels = load_libsvm_dense(path, 3, expected_rows=2)
        self.assertTrue(torch.equal(labels, torch.tensor([1.0, -1.0])))
        problem = CappedL1SVM(features, labels, cap=2.0, lam=0.1, device="cpu")
        self.assertTrue(
            torch.allclose(problem.features.norm(dim=1), torch.ones(2))
        )

    def test_component_loss_matches_hinge_plus_capped_penalty(self):
        features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        labels = torch.tensor([1.0, -1.0])
        problem = CappedL1SVM(
            features, labels, cap=0.5, lam=0.2, device="cpu", normalize_rows=False
        )
        point = torch.tensor([2.0, 1.0])
        self.assertTrue(
            torch.allclose(problem.component_losses(point), torch.tensor([0.2, 2.2]))
        )
        self.assertAlmostEqual(problem.accuracy(point), 0.5)

    def test_paired_points_match_zo_estimator_interface(self):
        problem = CappedL1SVM(
            torch.eye(2),
            torch.tensor([1.0, -1.0]),
            cap=2.0,
            lam=0.0,
            device="cpu",
            normalize_rows=False,
        )
        points = torch.tensor([[0.2, 0.0], [0.0, -0.2]])
        losses = problem.component_losses(points, torch.tensor([0, 1]))
        self.assertEqual(tuple(losses.shape), (2,))
        self.assertTrue(torch.isfinite(losses).all())

    def test_test_accuracy_uses_separate_normalized_split(self):
        problem = CappedL1SVM(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([1.0, -1.0]),
            cap=2.0,
            lam=0.0,
            device="cpu",
            test_features=torch.tensor([[3.0, 4.0], [-3.0, -4.0]]),
            test_labels=torch.tensor([1.0, -1.0]),
        )
        self.assertTrue(problem.has_test_data)
        self.assertTrue(
            torch.allclose(problem.test_features.norm(dim=1), torch.ones(2))
        )
        self.assertAlmostEqual(
            problem.test_accuracy(torch.tensor([1.0, 0.0])), 1.0
        )


if __name__ == "__main__":
    unittest.main()
