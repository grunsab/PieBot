import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from training.nnue import process_bins


class _CountingZstd:
    def __init__(self) -> None:
        self.stream_reader_calls = 0

    def ZstdDecompressor(self):
        owner = self

        class _Decompressor:
            def stream_reader(self, stream):
                owner.stream_reader_calls += 1
                return io.BytesIO(stream.read())

        return _Decompressor()


class DirectZstdIngestTests(unittest.TestCase):
    def test_direct_bin_zst_is_decompressed_exactly_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "sample.bin.zst"
            source.write_bytes(b"already-valid-after-one-decompression")
            writer = process_bins.ShardWriter(root / "out", shard_size=10)
            fake_zstd = _CountingZstd()
            try:
                with mock.patch.object(process_bins, "zstd", fake_zstd):
                    with mock.patch.object(
                        process_bins.lc0_bin,
                        "iter_v6_records",
                        return_value=iter(()),
                    ):
                        processed = process_bins.process_single_path(
                            source,
                            writer,
                            top_policy=1,
                            remaining=None,
                        )
            finally:
                writer.close()

            self.assertEqual(0, processed)
            self.assertEqual(1, fake_zstd.stream_reader_calls)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
