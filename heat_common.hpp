#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace heat {

constexpr float kHot = 100.0f;
constexpr float kCold = 0.0f;

struct CpuResult {
    int steps = 0;
    double elapsed_ms = 0.0;
    double final_max_delta = 0.0;
    bool converged = false;
};

inline std::size_t idx(int row, int col, int n) {
    return static_cast<std::size_t>(row) * static_cast<std::size_t>(n) +
           static_cast<std::size_t>(col);
}

inline void initialize_grid(std::vector<float>& grid, int n) {
    if (n < 3) {
        throw std::invalid_argument("N must be at least 3");
    }

    grid.assign(static_cast<std::size_t>(n) * static_cast<std::size_t>(n),
                kCold);

    for (int col = 0; col < n; ++col) {
        grid[idx(0, col, n)] = kHot;
        grid[idx(n - 1, col, n)] = kCold;
    }
    for (int row = 0; row < n; ++row) {
        grid[idx(row, 0, n)] = kCold;
        grid[idx(row, n - 1, n)] = kCold;
    }

    // The top corners belong to both a hot and a cold boundary.
    grid[idx(0, 0, n)] = 0.5f * (kHot + kCold);
    grid[idx(0, n - 1, n)] = 0.5f * (kHot + kCold);
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

inline void write_csv_grid(const std::string& path,
                           const std::vector<float>& grid,
                           int n) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open output file: " + path);
    }

    out << std::setprecision(8);
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            if (col > 0) {
                out << ',';
            }
            out << grid[idx(row, col, n)];
        }
        out << '\n';
    }
}

inline double max_abs_difference(const std::vector<float>& a,
                                 const std::vector<float>& b) {
    double max_diff = 0.0;
    for (std::size_t k = 0; k < a.size(); ++k) {
        const double diff = std::fabs(static_cast<double>(a[k] - b[k]));
        max_diff = std::max(max_diff, diff);
    }
    return max_diff;
}

inline double throughput_gpts(int n, int steps, double elapsed_ms) {
    const double points =
        static_cast<double>(n - 2) * static_cast<double>(n - 2) * steps;
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

    auto save_snapshot = [&](int step) {
        if (snapshot_prefix.empty()) {
            return;
        }
        if (std::binary_search(snapshots.begin(), snapshots.end(), step)) {
            std::ostringstream path;
            path << snapshot_prefix << "_step_" << step << ".csv";
            write_csv_grid(path.str(), current, n);
        }
    };

    CpuResult result;
    save_snapshot(0);

    const auto start = std::chrono::steady_clock::now();
    for (int step = 0; step < max_steps; ++step) {
        double max_delta = 0.0;

        for (int row = 1; row < n - 1; ++row) {
            for (int col = 1; col < n - 1; ++col) {
                const std::size_t k = idx(row, col, n);
                const float updated =
                    current[k] +
                    r * (current[k - n] + current[k + n] + current[k - 1] +
                         current[k + 1] - 4.0f * current[k]);

                next[k] = updated;
                max_delta = std::max(
                    max_delta,
                    std::fabs(static_cast<double>(updated - current[k])));
            }
        }

        current.swap(next);
        result.steps = step + 1;
        result.final_max_delta = max_delta;
        save_snapshot(result.steps);

        if (eps > 0.0f && max_delta < eps) {
            result.converged = true;
            break;
        }
    }

    const auto stop = std::chrono::steady_clock::now();
    result.elapsed_ms =
        std::chrono::duration<double, std::milli>(stop - start).count();
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
