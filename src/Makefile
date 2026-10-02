AVX_ON ?= 0

CXX = g++
CXXFLAGS ?= -std=c++23 -march=native -O3 -Wall -Wextra -pedantic
PYTHON ?= python3
AVXFLAGS = -mavx

SRC = main.cpp
OUT = gemm
HEADERS = GEMM.h Timer.h Utils.h
TEST_OUT = test_gemm

ifeq ($(AVX_ON), 1)
    TARGET_FLAGS = $(AVXFLAGS)
endif

.PHONY: all clean test

all: $(OUT)

$(OUT): $(SRC) $(HEADERS) Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(TARGET_FLAGS) $(SRC) $(LDFLAGS) $(LDLIBS) -o $@

$(TEST_OUT): tests/test_gemm.cpp $(HEADERS) Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(TARGET_FLAGS) -I. tests/test_gemm.cpp $(LDFLAGS) $(LDLIBS) -o $@

test: $(OUT) $(TEST_OUT)
	./$(TEST_OUT)
	$(PYTHON) tests/test_cli.py

clean:
	rm -f $(OUT) $(TEST_OUT)
