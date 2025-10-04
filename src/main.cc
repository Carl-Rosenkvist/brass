#include <iostream>
#include <string>
#include <vector>
#include <filesystem>

#include "analysis.h"          // run_analysis(...)
                                // parse_merge_key is called inside run_analysis
#include "analysisregister.h"  // for list_registered()

int main(int argc, char* argv[]) {
    if (argc > 1 && std::string(argv[1]) == "--list-analyses") {
        const auto registered = AnalysisRegistry::instance().list_registered();
        std::cout << "Available analyses:\n";
        for (const auto& name : registered) {
            std::cout << " - " << name << "\n";
        }
        return 0;
    }

    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <file[:key=val,...]>... <analysis> <quantities...>"
                  << " [--output-folder <path>]\n"
                  << "       or: " << argv[0] << " --list-analyses\n";
        return 1;
    }

    // Collect (file, meta) pairs — meta is the substring after ':' (or empty)
    std::vector<std::pair<std::string, std::string>> file_and_meta;
    int i = 1;
    for (; i < argc; ++i) {
        std::string arg = argv[i];
        // treat anything with ":" or ending in ".bin" as an input spec
        if (ends_with(arg, ".bin") || arg.find(':') != std::string::npos) {
            auto pos = arg.find(':');
            if (pos == std::string::npos) {
                file_and_meta.emplace_back(arg, std::string{});
            } else {
                file_and_meta.emplace_back(arg.substr(0, pos), arg.substr(pos + 1));
            }
        } else {
            break; // next token should be the analysis name
        }
    }

    if (i >= argc) {
        std::cerr << "Error: No analysis specified.\n";
        return 1;
    }
    const std::string analysis_name = argv[i++];

    // Output folder and quantities
    std::filesystem::path output_folder = ".";
    std::vector<std::string> quantities;

    for (; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--output-folder") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --output-folder requires a path argument.\n";
                return 1;
            }
            output_folder = argv[++i];
        } else {
            quantities.push_back(std::move(arg));
        }
    }

    try {
        run_analysis(file_and_meta,
                     analysis_name,
                     quantities,
                     output_folder.string());
    } catch (const std::exception& e) {
        std::cerr << "run_analysis failed: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
