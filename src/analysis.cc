#include "analysis.h"

#include <yaml-cpp/yaml.h>

#include "analysisregister.h"

void Analysis::on_header(Header& header) {
    smash_version = header.smash_version;
}

// DispatchingAccessor methods
void DispatchingAccessor::register_analysis(
    std::shared_ptr<Analysis> analysis) {
    analyses.push_back(std::move(analysis));
}
// DispatchingAccessor methods
void DispatchingAccessor::create_and_register_analysis(
    const std::string& analysis_name) {
    analyses.push_back(AnalysisRegistry::instance().create(analysis_name));
}

void DispatchingAccessor::on_particle_block(const ParticleBlock& block) {
    for (auto& a : analyses) {
        a->analyze_particle_block(block, *this);
    }
}
void DispatchingAccessor::on_interaction_block(const InteractionBlock& block) {
    for (auto& a : analyses) {
        a->analyze_interaction_block(block, *this);
    }
}

void DispatchingAccessor::on_end_block(const EndBlock& block) {
    for (auto& a : analyses) {
        a->analyze_end_block(block, *this);
    }
}

void DispatchingAccessor::on_header(Header& header) {
    for (auto& a : analyses) {
        a->on_header(header);
    }
}
