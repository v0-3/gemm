import argparse


def positive_dimension(value):
    try:
        dimension = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("dimensions must be positive 32-bit integers") from None
    if not 0 < dimension <= 2**31 - 1:
        raise argparse.ArgumentTypeError("dimensions must be positive 32-bit integers")
    return dimension


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", dest="M", type=positive_dimension, help="Number of rows for matrix A", required=True
    )
    parser.add_argument(
        "-n", dest="N", type=positive_dimension, help="Number of columns for matrix B", required=True
    )
    parser.add_argument(
        "-k",
        dest="K",
        type=positive_dimension,
        help="Number of columns/rows for matrices A & B",
        required=True,
    )
    args = parser.parse_args()

    try:
        import numpy as np
    except ImportError:
        parser.error("NumPy is required to generate matrices; install it with python3 -m pip install numpy")

    try:
        A = np.random.randn(args.M, args.K).astype(np.float32)
        B = np.random.randn(args.K, args.N).astype(np.float32)
        C = np.dot(A, B)

        with open("data.bin", "wb") as f:
            f.write(A.data)
            f.write(B.data)
            f.write(C.data)
    except (MemoryError, OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
