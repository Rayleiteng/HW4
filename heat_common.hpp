#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace heat {

constexpr float kTopTemperature = 100.0f;
constexpr float kColdTemperature = 0.0f;

struct CpuResult {
    int steps = 0;
    double elapsed_ms = 0.0;
    double final_max_delta = 0.0;
    bool converged = false;
};

inline std::size_t idx(int i, int j, int n) {
    return static_cast<std::size_t>(i) * static_cast<std::size_t>(n) +
           static_cast<std::size_t>(j);
}

inline void initialize_grid(std::vector<float>& grid, int n) {
    if (n < 3) {
        throw std::invalid_argument("N must be at least 3");
    }

    grid.assign(static_cast<std::size_t>(n) * static_cast<std::size_t>(n),
                kColdTemperature);

    for (int j = 0; j < n; ++j) {
        grid[idx(0, j, n)] = kTopTemperature;
        grid[idx(n - 1, j, n)] = kColdTemperature;
    }
    for (int i = 0; i < n; ++i) {
        grid[idx(i, 0, n)] = kColdTemperature;
        grid[idx(i, n - 1, n)] = kColdTemperature;
    }

    grid[idx(0, 0, n)] = 0.5f * (kTopTemperature + kColdTemperature);
    grid[idx(0, n - 1, n)] = 0.5f * (kTopTemperature + kColdTemperature);
}

inline double max_abs_difference(const std::vector<float>& a,
                                 const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("grid sizes differ");
    }
    double max_diff = 0.0;
    for (std::size_t k = 0; k < a.size(); ++k) {
        max_diff = std::max(max_diff,
                            static_cast<double>(std::fabs(a[k] - b[k])));
    }
    return max_diff;
}

inline void write_csv_grid(const std::string& path,
                           const std::vector<float>& grid,
                           int n) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open output file: " + path);
    }
    out << std::setprecision(8);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (j != 0) {
                out << ',';
            }
            out << grid[idx(i, j, n)];
        }
        out << '\n';
    }
}

inline std::vector<int> parse_int_list(const std::string& text) {
    std::vector<int> values;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            values.push_back(std::stoi(item));
        }
    }
    return values;
}

inline bool has_arg(int argc, char** argv, const std::string& name) {
    for (int i = 1; i < argc; ++i) {
        if (argv[i] == name) {
            return true;
        }
    }
    return false;
}

inline std::string get_arg(int argc,
                           char** argv,
                           const std::string& name,
                           const std::string& default_value) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return argv[i + 1];
        }
    }
    return default_value;
}

inline int get_int_arg(int argc, char** argv, const std::string& name, int value) {
    return std::stoi(get_arg(argc, argv, name, std::to_string(value)));
}

inline float get_float_arg(int argc,
                           char** argv,
                           const std::string& name,
                           float value) {
    return std::stof(get_arg(argc, argv, name, std::to_string(value)));
}

inline double throughput_gpts(int n, int steps, double elapsed_ms) {
    const double points = static_cast<double>(n - 2) * static_cast<double>(n - 2) *
                          static_cast<double>(steps);
    return points / (elapsed_ms / 1000.0) / 1.0e9;
}

inline CpuResult solve_cpu(std::vector<float>& current,
                           int n,
                           int max_steps,
                           float r,
                           float eps,
                           const std::vector<int>& snapshot_steps = {},
                           const std::string& snapshot_prefix = "") {
    std::vector<float> next = current;
    std::vector<int> snapshots = snapshot_steps;
    std::sort(snapshots.begin(), snapshots.end());
    snapshots.erase(std::unique(snapshots.begin(), snapshots.end()),
                    snapshots.end());

    auto maybe_write_snapshot = [&](int step) {
        if (snapshot_prefix.empty()) {
            return;
        }
        if (std::binary_search(snapshots.begin(), snapshots.end(), step)) {
            std::ostringstream name;
            name << snapshot_prefix << "_step_" << step << ".csv";
            write_csv_grid(name.str(), current, n);
        }
    };

    maybe_write_snapshot(0);
    CpuResult result;
    const auto t0 = std::chrono::steady_clock::now();

    for (int step = 0; step < max_steps; ++step) {
        double max_delta = 0.0;
        for (int i = 1; i < n - 1; ++i) {
            const std::size_t row = static_cast<std::size_t>(i) *
                                    static_cast<std::size_t>(n);
            for (int j = 1; j < n - 1; ++j) {
                const std::size_t k = row + static_cast<std::size_t>(j);
                const float updated =
                    current[k] +
                    r * (current[k - static_cast<std::size_t>(n)] +
                         current[k + static_cast<std::size_t>(n)] +
                         current[k - 1] + current[k + 1] -
                         4.0f * current[k]);
                next[k] = updated;
                max_delta = std::max(
                    max_delta,
                    static_cast<double>(std::fabs(updated - current[k])));
            }
        }

        current.swap(next);
        result.steps = step + 1;
        result.final_max_delta = max_delta;
        maybe_write_snapshot(result.steps);

        if (eps > 0.0f && max_delta < static_cast<double>(eps)) {
            result.converged = true;
            break;
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    result.elapsed_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    return result;
}

inline void print_result_line(const std::string& mode,
                              int n,
                              int steps,
                              double elapsed_ms,
                              double final_delta,
                              bool converged,
                              const std::string& extra = "") {
    std::cout << "RESULT"
              << ",mode=" << mode
              << ",N=" << n
              << ",steps=" << steps
              << ",elapsed_ms=" << std::fixed << std::setprecision(3)
              << elapsed_ms
              << ",throughput_Gpt_s=" << std::setprecision(6)
              << throughput_gpts(n, steps, elapsed_ms)
              << ",max_delta=" << std::scientific << std::setprecision(6)
              << final_delta
              << ",converged=" << (converged ? 1 : 0);
    if (!extra.empty()) {
        std::cout << ',' << extra;
    }
    std::cout << '\n';
}

}  // namespace heat
