#pragma once

#include <chrono>
#include <stdexcept>

class Timer {
   private:
    bool status = false;
    bool completed = false;
    std::chrono::steady_clock::time_point t1{};
    std::chrono::steady_clock::time_point t2{};

   public:
    void start() {
        t1 = std::chrono::steady_clock::now();
        status = true;
        completed = false;
    }

    void stop() {
        if (!status) {
            throw std::runtime_error("Timer has not been started!");
        }
        t2 = std::chrono::steady_clock::now();
        status = false;
        completed = true;
    }

    double duration() const {
        if (!completed) {
            throw std::runtime_error("Timer has not been stopped!");
        }
        return std::chrono::duration<double, std::milli>(t2 - t1).count();
    }
};
