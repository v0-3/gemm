#pragma once

#include <cmath>
#include <cstddef>
#include <print>
#include <sstream>
#include <stdexcept>
#include <string>

inline bool is_equal(const float *m1, const float *m2, const int rows, const int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            const std::size_t index = static_cast<std::size_t>(i) * cols + j;
            // Allow float accumulation roundoff relative to the reference magnitude.
            const float tolerance = 1.0e-3f + 1.0e-5f * std::fabs(m1[index]);
            if (!std::isfinite(m1[index]) || !std::isfinite(m2[index]) ||
                std::fabs(m1[index] - m2[index]) > tolerance) {
                std::print(stderr, "\n[ERROR] Mismatch at ({}, {}) m1[{}] != m2[{}]\n", i, j, m1[index], m2[index]);
                return false;
            }
        }
    }

    std::print("\n[SUCCESS] All done!\n");
    return true;
}

inline void print_matrix(const float *m, const int rows, const int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::print("{} ", m[static_cast<std::size_t>(i) * cols + j]);
        }
        std::print("\n");
    }
    std::print("\n\n");
}

inline int get_arg_value(const char *arg) {
    int n = 0;
    std::istringstream ss(arg);
    if (!(ss >> n)) {
        throw std::invalid_argument(std::string("Invalid value: ") + arg);
    } else if (!ss.eof()) {
        throw std::invalid_argument(std::string("Trailing characters after number: ") + arg);
    }

    if (n <= 0) {
        throw std::invalid_argument(std::string("Invalid value: ") + arg);
    }

    return n;
}
