import unittest

try:
    from training.nnue import train_torch
except Exception:  # pragma: no cover - exercised on environments without torch
    train_torch = None  # type: ignore[assignment]


@unittest.skipUnless(
    train_torch is not None and train_torch.torch_available(),
    "torch is not installed",
)
class TorchBatchPackingTests(unittest.TestCase):
    def test_all_empty_feature_bags_preserve_batch_shape(self) -> None:
        device = train_torch.torch.device("cpu")
        flat, offsets, targets = train_torch._pack_batch(
            [[], [], []],
            [1.0, 0.0, -1.0],
            device,
        )

        self.assertEqual(0, flat.numel())
        self.assertEqual([0, 0, 0], offsets.tolist())
        self.assertEqual((3,), tuple(targets.shape))

        model = train_torch.TorchNnue(input_dim=4, hidden_dim=2)
        predictions = model(flat, offsets)
        self.assertEqual((3,), tuple(predictions.shape))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
