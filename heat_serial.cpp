#include "heat_common.hpp"

#include <exception>

namespace {

void usage(const char* program) {
    std::cerr
        << "Usage: " << program << " [options]\n"
        << "  --n N                       grid size, default 1024\n"
        << "  --steps STEPS               maximum time steps, default 2000\n"
        << "  --r R                       alpha*dt/h^2, default 0.24\n"
        << "  --eps EPS                   stop if max update is below EPS; 0 disables\n"
        << "  --output FILE               write final grid CSV\n"
        << "  --snapshot-prefix PREFIX    write selected snapshot CSV files\n"
        << "  --snapshot-steps LIST       comma separated steps, e.g. 0,100,500,2000\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (heat::has_arg(argc, argv, "--help")) {
            usage(argv[0]);
            return 0;
        }

        const int n = heat::get_int_arg(argc, argv, "--n", 1024);
        const int steps = heat::get_int_arg(argc, argv, "--steps", 2000);
        const float r = heat::get_float_arg(argc, argv, "--r", 0.24f);
        const float eps = heat::get_float_arg(argc, argv, "--eps", 0.0f);
        const std::string output =
            heat::get_arg(argc, argv, "--output", std::string());
        const std::string snapshot_prefix =
            heat::get_arg(argc, argv, "--snapshot-prefix", std::string());
        const std::string snapshot_text =
            heat::get_arg(argc, argv, "--snapshot-steps", std::string());
        const std::vector<int> snapshots =
            snapshot_text.empty() ? std::vector<int>{}
                                  : heat::parse_int_list(snapshot_text);

        if (r <= 0.0f || r > 0.25f) {
            throw std::invalid_argument("r must be in (0, 0.25] for stability");
        }

        std::vector<float> grid;
        heat::initialize_grid(grid, n);
        const heat::CpuResult result =
            heat::solve_cpu(grid, n, steps, r, eps, snapshots, snapshot_prefix);

        if (!output.empty()) {
            heat::write_csv_grid(output, grid, n);
        }

        heat::print_result_line("serial",
                                n,
                                result.steps,
                                result.elapsed_ms,
                                result.final_max_delta,
                                result.converged);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        usage(argv[0]);
        return 1;
    }
}
