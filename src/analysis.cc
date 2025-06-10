#include "analysis.h"
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include "analysisregister.h"


void Analysis::set_merge_keys(MergeKeySet k) {
    keys = std::move(k);
}

const MergeKeySet& Analysis::get_merge_keys() const {
    return keys;
}

void Analysis::on_header(Header& header) {
    smash_version = header.smash_version;
}

// DispatchingAccessor methods
void DispatchingAccessor::register_analysis(std::shared_ptr<Analysis> analysis) {
    analyses.push_back(std::move(analysis));
}

void DispatchingAccessor::on_particle_block(const ParticleBlock& block) {
    for (auto& a : analyses) {
        a->analyze_particle_block(block, *this);
    }
}

void DispatchingAccessor::on_end_block(const EndBlock&) {
    // optional
}

void DispatchingAccessor::on_header(Header& header) {
    for (auto& a : analyses) {
        a->on_header(header);
    }
}

void run_analysis(const std::vector<std::pair<std::string, std::string>>& file_and_meta,
                  const std::string& analysis_name,
                  const std::vector<std::string>& quantities,
                  bool save_output,
                  bool print_output,
                  const std::string& output_folder)
{
    if (quantities.empty()) throw std::runtime_error("No quantities provided");

    if (save_output) {
        std::error_code ec;
        std::filesystem::create_directories(output_folder, ec);
        if (ec) throw std::runtime_error("create_directories failed: " + ec.message());
    }

    std::vector<std::pair<std::string, MergeKeySet>> input_files;
    input_files.reserve(file_and_meta.size());
    for (const auto& [file, meta] : file_and_meta) {
        MergeKeySet ks = parse_merge_key(meta);
        sort_keyset(ks);
        input_files.emplace_back(file, std::move(ks));
    }

    std::vector<Entry> results;
    results.reserve(input_files.size());

    auto find_or_insert = [&](MergeKeySet k) -> std::shared_ptr<Analysis>& {
        auto it = std::lower_bound(results.begin(), results.end(), k,
            [](Entry const& e, MergeKeySet const& x){ return e.key < x; });
        if (it == results.end() || it->key < k || k < it->key) {
            it = results.insert(it, Entry{std::move(k), nullptr});
        }
        return it->analysis;
    };

    for (auto& [path, key] : input_files) {
        auto analysis = AnalysisRegistry::instance().create(analysis_name);
        if (!analysis) throw std::runtime_error("Unknown analysis: " + analysis_name);

        analysis->set_merge_keys(key);

        auto dispatcher = std::make_shared<DispatchingAccessor>();
        dispatcher->register_analysis(analysis);

        BinaryReader reader(path, quantities, dispatcher);
        reader.read();

        auto& slot = find_or_insert(std::move(key));
        if (slot) {
            *slot += *analysis;
        } else {
            slot = std::move(analysis);
        }
    }

    for (auto& e : results) {
        e.analysis->finalize();
        
    if (save_output) {
        e.analysis->save(output_folder);
    }
}
}




