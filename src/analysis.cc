#include "analysis.h"
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include "analysisregister.h"




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
                  const std::vector<std::string>& analysis_names,
                  const std::vector<std::string>& quantities,
                  const std::string& output_folder)
{
    if (quantities.empty()) throw std::runtime_error("No quantities provided");

    std::error_code ec;
    std::filesystem::create_directories(output_folder, ec);
    if (ec) throw std::runtime_error("create_directories failed: " + ec.message());

    // Parse & sort merge keys once
    std::vector<std::pair<std::string, MergeKeySet>> inputs;
    inputs.reserve(file_and_meta.size());
    for (const auto& [file, meta] : file_and_meta) {
        MergeKeySet ks = parse_merge_key(meta);
        sort_keyset(ks);
        inputs.emplace_back(file, std::move(ks));
    }

    // Run each file with its own analysis instance
    std::vector<std::shared_ptr<Analysis>> analyses;

    for (auto& [path, key] : inputs) {
        
        auto dispatcher = std::make_shared<DispatchingAccessor>();
        for(const auto& analysis_name : analysis_names){
            auto analysis = AnalysisRegistry::instance().create(analysis_name);
            analysis->set_merge_keys(key);
        
        analyses.push_back(std::move(analysis));
        dispatcher->register_analysis(analyses.back());
        }
        BinaryReader reader(path, quantities, dispatcher);
        reader.read();

    }

    // Sort by (analysis name, merge key) to make reduction linear
    std::sort(analyses.begin(), analyses.end(),
        [](const std::shared_ptr<Analysis>& a, const std::shared_ptr<Analysis>& b){
            if (a->name() != b->name()) return a->name() < b->name();
            const auto& ka = a->get_merge_keys();
            const auto& kb = b->get_merge_keys();
            if (ka < kb) return true;
            if (kb < ka) return false;
            return false;
        });

    // Reduce with can_combine + operator+=
    std::vector<std::shared_ptr<Analysis>> reduced;
    reduced.reserve(analyses.size());
    for (auto& cur : analyses) {
        if (reduced.empty() || !reduced.back()->can_combine(*cur)) {
            reduced.push_back(std::move(cur));
        } else {
            *reduced.back() += *cur;
        }
    }

    // Finalize & save
    for (auto& a : reduced) {
        a->finalize();
        a->save(output_folder);
    }
}




