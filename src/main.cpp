#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <print>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "GEMM.h"
#include "Timer.h"
#include "Utils.h"

int main(int argc, char *argv[]) try {
    if (argc != 7) {
        std::print("Usage:\n\tgemm -m <M> -n <N> -k <K>\n\n");
        std::print("Options:\n");
        std::print("  -m M        Number of rows for matrix A\n");
        std::print("  -n N        Number of columns for matrix B\n");
        std::print("  -k K        Number of columns/rows for matrices A & B\n");
        return EXIT_FAILURE;
    }

    if (std::string(argv[1]) != "-m" || std::string(argv[3]) != "-n" || std::string(argv[5]) != "-k") {
        throw std::invalid_argument("Invalid flags");
    }

    const int M = get_arg_value(argv[2]);  // rows of matrix A
    const int N = get_arg_value(argv[4]);  // columns of matrix B
    const int K = get_arg_value(argv[6]);  // number of columns/rows A&B

    constexpr std::uintmax_t max_elements =
        std::min<std::uintmax_t>(std::numeric_limits<std::size_t>::max(),
                                 std::numeric_limits<std::streamsize>::max()) / sizeof(float);
    const auto matrix_size = [](int rows, int cols) {
        if (static_cast<std::uintmax_t>(rows) > max_elements / static_cast<std::uintmax_t>(cols)) {
            throw std::length_error("Matrix dimensions are too large");
        }
        return static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    };
    const std::size_t a_size = matrix_size(M, K);
    const std::size_t b_size = matrix_size(K, N);
    const std::size_t c_size = matrix_size(M, N);
    std::size_t total_elements = 0;
    for (std::size_t size : {a_size, b_size, c_size}) {
        if (size > max_elements - total_elements) {
            throw std::length_error("Matrix dimensions are too large");
        }
        total_elements += size;
    }
    const auto expected_bytes = static_cast<std::streamsize>(total_elements * sizeof(float));

    std::ifstream input("data.bin", std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("Unable to open data.bin, please use gemm.py to generate data file!");
    }
    if (input.tellg() != expected_bytes) {
        throw std::runtime_error("data.bin size does not match the requested matrix dimensions");
    }
    input.seekg(0);

    std::vector<float> A(a_size), B(b_size), C(c_size), vals(c_size);
    const auto read_matrix = [&input](std::vector<float> &matrix) {
        if (!input.read(reinterpret_cast<char *>(matrix.data()),
                        static_cast<std::streamsize>(matrix.size() * sizeof(float)))) {
            throw std::runtime_error("Unable to read matrix data from data.bin");
        }
    };
    read_matrix(A);
    read_matrix(B);
    read_matrix(C);
    input.close();

    int opt = 0;
    std::print("Select Matrix Multiplication Method\n{:=>35}\n", '=');
    std::print("[1] Naive GEMM\n");
    std::print("[2] Loop order GEMM\n");
    std::print("[3] Tiling GEMM\n");
#if defined(__AVX__)
    std::print("[4] AVX GEMM\n> ");
#else
    std::print("[4] AVX GEMM (unavailable in this build)\n> ");
#endif
    std::string selection;
    if (!std::getline(std::cin, selection)) {
        throw std::invalid_argument("Invalid method selected!");
    }
    std::istringstream method_input(selection);
    if (!(method_input >> opt) || !(method_input >> std::ws).eof()) {
        throw std::invalid_argument("Invalid method selected!");
    }

    using Kernel = void (*)(const float *, const float *, float *, int, int, int);
    Kernel multiply = nullptr;
    switch (opt) {
        case 1:
            multiply = naive;
            break;
        case 2:
            multiply = looporder;
            break;
        case 3:
            multiply = tiling;
            break;
        case 4:
#if defined(__AVX__)
            multiply = AVX;
#else
            throw std::runtime_error("AVX is unavailable in this build!");
#endif
            break;
        default:
            throw std::invalid_argument("Invalid method selected!");
    }

    Timer t;
    t.start();
    multiply(A.data(), B.data(), vals.data(), M, N, K);
    t.stop();
    const double milliseconds = t.duration();
    const double gflops = milliseconds > 0.0 ? (2.0 * M * N * K) / (milliseconds * 1.0e6) : 0.0;

    std::print("\nResults\n{:=>35}\n", '=');
    std::print("\tA Matrix [{} x {}]\n", M, K);
    std::print("\tB Matrix [{} x {}]\n", K, N);
    std::print("\tC Matrix [{} x {}]\n\n", M, N);
    std::print("\tM         = {:10}\n", M);
    std::print("\tN         = {:10}\n", N);
    std::print("\tK         = {:10}\n", K);
    std::print("\tGFLOPS    = {:10.5f}\n", gflops);
    std::print("\tTime (ms) = {:10.6f}\n", milliseconds);

    return is_equal(C.data(), vals.data(), M, N) ? EXIT_SUCCESS : EXIT_FAILURE;
} catch (const std::exception &error) {
    std::print(stderr, "[ERROR] {}\n", error.what());
    return EXIT_FAILURE;
}
