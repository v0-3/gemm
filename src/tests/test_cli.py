import importlib.util
import math
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1]
EXECUTABLE = SOURCE / "gemm"
GENERATOR = SOURCE / "gemm.py"


def matrix_data(m, n, k):
    a = [(i % 7 - 3) * 0.25 for i in range(m * k)]
    b = [(i % 11 - 5) * 0.5 for i in range(k * n)]
    c = [sum(a[i * k + p] * b[p * n + j] for p in range(k))
         for i in range(m) for j in range(n)]
    values = a + b + c
    return struct.pack(f"={len(values)}f", *values)


class CommandLineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.cwd = Path(self.directory.name)

    def run_gemm(self, m=2, n=3, k=5, method="1\n", data=None, args=None):
        if data is not None:
            (self.cwd / "data.bin").write_bytes(data)
        if args is None:
            args = ["-m", str(m), "-n", str(n), "-k", str(k)]
        return subprocess.run([str(EXECUTABLE), *args], input=method, text=True,
                              capture_output=True, cwd=self.cwd, timeout=10)

    def assert_failure(self, result):
        output = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0, output)
        self.assertIn("[ERROR]", output)
        self.assertNotIn("[SUCCESS]", output)

    def test_scalar_methods_and_finite_timing(self):
        for m, n, k in [(1, 1, 1), (2, 9, 5), (5, 3, 7)]:
            for method in (1, 2, 3):
                with self.subTest(shape=(m, n, k), method=method):
                    result = self.run_gemm(m, n, k, f"{method}\n", matrix_data(m, n, k))
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn("[SUCCESS]", result.stdout)
                    value = re.search(r"GFLOPS\s*=\s*(\S+)", result.stdout)
                    self.assertIsNotNone(value)
                    self.assertTrue(math.isfinite(float(value.group(1))))

    def test_avx_or_explicit_unavailability(self):
        for n in (1, 7, 8, 9, 15, 16, 17):
            with self.subTest(columns=n):
                result = self.run_gemm(3, n, 5, "4\n", matrix_data(3, n, 5))
                if "[4] AVX GEMM (unavailable" in result.stdout:
                    self.assert_failure(result)
                    self.assertNotIn("Results", result.stdout)
                else:
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn("[SUCCESS]", result.stdout)

    def test_apple_silicon_or_explicit_unavailability(self):
        for m, n, k in [(1, 1, 1), (5, 7, 9), (16, 16, 16), (17, 19, 23), (65, 31, 33)]:
            with self.subTest(shape=(m, n, k)):
                result = self.run_gemm(m, n, k, "5\n", matrix_data(m, n, k))
                if "[5] Apple Silicon GEMM (unavailable" in result.stdout:
                    self.assert_failure(result)
                    self.assertNotIn("Results", result.stdout)
                else:
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn("[SUCCESS]", result.stdout)

    def test_reference_mismatch(self):
        data = bytearray(matrix_data(2, 3, 5))
        struct.pack_into("=f", data, len(data) - 4, 999.0)
        self.assert_failure(self.run_gemm(data=data))

    def test_nonfinite_data(self):
        for offset in (0, -4):
            for value in (math.nan, math.inf, -math.inf):
                with self.subTest(offset=offset, value=value):
                    data = bytearray(matrix_data(2, 3, 5))
                    struct.pack_into("=f", data, offset, value)
                    self.assert_failure(self.run_gemm(data=data))

    def test_missing_file(self):
        self.assert_failure(self.run_gemm())

    def test_wrong_file_size(self):
        valid = matrix_data(2, 3, 5)
        for data in (b"", valid[:4], valid[:-1], valid + b"x"):
            with self.subTest(length=len(data)):
                result = self.run_gemm(data=data)
                self.assert_failure(result)
                self.assertNotIn("Select Matrix", result.stdout)

    def test_invalid_method(self):
        for method in ("", "0\n", "6\n", "abc\n", "1x\n", "1 2\n"):
            with self.subTest(method=method):
                result = self.run_gemm(method=method, data=matrix_data(2, 3, 5))
                self.assert_failure(result)
                self.assertNotIn("Results", result.stdout)

    def test_invalid_dimensions_and_flags(self):
        for value in ("0", "-1", "1x", "1.5", "2147483648"):
            with self.subTest(value=value):
                self.assert_failure(self.run_gemm(m=value))
        self.assert_failure(self.run_gemm(args=["-x", "1", "-n", "1", "-k", "1"]))
        self.assertNotEqual(self.run_gemm(args=[]).returncode, 0)

    def test_oversized_dimensions_fail_before_allocation(self):
        for dimension in (2147483647, 1_000_000_000):
            with self.subTest(dimension=dimension):
                result = self.run_gemm(dimension, dimension, dimension, data=b"")
                self.assert_failure(result)
                self.assertIn("Matrix dimensions are too large", result.stderr)
                self.assertNotIn("Select Matrix", result.stdout)

    def test_large_matrix_counts_do_not_overflow(self):
        result = self.run_gemm(50000, 50000, 1, data=b"")
        self.assert_failure(result)
        self.assertIn("data.bin size does not match", result.stderr)

    def run_generator(self, *args):
        return subprocess.run([sys.executable, str(GENERATOR), *args], text=True,
                              capture_output=True, cwd=self.cwd, timeout=30)

    def test_generator_help(self):
        result = self.run_generator("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Number of rows", result.stdout)

    def test_generator_missing_numpy(self):
        result = subprocess.run(
            [sys.executable, "-S", str(GENERATOR), "-m", "2", "-n", "3", "-k", "5"],
            text=True, capture_output=True, cwd=self.cwd, timeout=10,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("NumPy is required", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertFalse((self.cwd / "data.bin").exists())

    def test_generator_invalid_dimensions(self):
        for flag in ("-m", "-n", "-k"):
            for value in ("0", "-1", "1.5", "2147483648"):
                with self.subTest(flag=flag, value=value):
                    args = ["-m", "2", "-n", "3", "-k", "5"]
                    args[args.index(flag) + 1] = value
                    result = self.run_generator(*args)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("error:", result.stderr)
                    self.assertNotIn("Traceback", result.stderr)
                    self.assertFalse((self.cwd / "data.bin").exists())

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is not installed")
    def test_generator_round_trip(self):
        result = self.run_generator("-m", "3", "-n", "9", "-k", "5")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.cwd / "data.bin").stat().st_size, 4 * (3 * 5 + 5 * 9 + 3 * 9))
        for method in (1, 2, 3):
            result = self.run_gemm(3, 9, 5, f"{method}\n")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        result = self.run_gemm(3, 9, 5, "5\n")
        if "[5] Apple Silicon GEMM (unavailable" not in result.stdout:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
