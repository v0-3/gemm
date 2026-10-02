AVX_ON ?= 0

CXX = g++
CXXFLAGS ?= -std=c++23 -march=native -O3 -Wall -Wextra -pedantic
OBJCXX = clang++
OBJCXXFLAGS ?= $(CXXFLAGS)
PYTHON ?= python3
AVXFLAGS = -mavx

SRC = main.cpp
OUT = gemm
HEADERS = GEMM.h Timer.h Utils.h
TEST_OUT = test_gemm
BENCH_OUT = bench_gemm

ifeq ($(AVX_ON), 1)
    TARGET_FLAGS = $(AVXFLAGS)
endif

# Include the guarded Apple source for every Darwin target, including universal
# builds. Compile Objective-C++ with Clang even when CXX is GNU G++.
TARGET_TRIPLE := $(shell $(CXX) $(CPPFLAGS) $(CXXFLAGS) $(TARGET_FLAGS) -dumpmachine)
ifneq ($(findstring -apple-,$(TARGET_TRIPLE)),)
    APPLE_OBJ = AppleGEMM.o
    APPLE_LIBS = -framework Accelerate -framework Foundation -framework Metal -framework MetalPerformanceShaders
endif

.PHONY: all clean test bench

all: $(OUT)

AppleGEMM.o: AppleGEMM.mm GEMM.h Makefile
	$(OBJCXX) $(CPPFLAGS) $(OBJCXXFLAGS) $(TARGET_FLAGS) -fobjc-arc -c AppleGEMM.mm -o $@

$(OUT): $(SRC) $(APPLE_OBJ) $(HEADERS) Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(TARGET_FLAGS) $(SRC) $(APPLE_OBJ) $(LDFLAGS) $(LDLIBS) $(APPLE_LIBS) -o $@

$(TEST_OUT): tests/test_gemm.cpp $(APPLE_OBJ) $(HEADERS) Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(TARGET_FLAGS) -I. tests/test_gemm.cpp $(APPLE_OBJ) $(LDFLAGS) $(LDLIBS) $(APPLE_LIBS) -o $@

$(BENCH_OUT): tests/bench_gemm.cpp $(APPLE_OBJ) $(HEADERS) Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(TARGET_FLAGS) -I. tests/bench_gemm.cpp $(APPLE_OBJ) $(LDFLAGS) $(LDLIBS) $(APPLE_LIBS) -o $@

bench: $(BENCH_OUT)
	./$(BENCH_OUT)

test: $(OUT) $(TEST_OUT)
	./$(TEST_OUT)
	$(PYTHON) tests/test_cli.py

clean:
	rm -f $(OUT) $(TEST_OUT) $(BENCH_OUT) $(APPLE_OBJ)
