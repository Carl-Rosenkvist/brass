// src/analysis_registry.cc
#include "analysisregister.h"

#include <stdexcept>

AnalysisRegistry &AnalysisRegistry::instance() {
    static AnalysisRegistry registry;
    return registry;
}

void AnalysisRegistry::register_factory(const std::string &name,
                                        Factory factory) {
    factories_[name] = std::move(factory);
}

std::shared_ptr<Analysis> AnalysisRegistry::create(
    const std::string &name) const {
    auto it = factories_.find(name);
    if (it == factories_.end())
        throw std::runtime_error("No such analysis: " + name);
    return it->second();
}

std::vector<std::string> AnalysisRegistry::list_registered() const {
    std::vector<std::string> keys;
    for (const auto &[k, _] : factories_) keys.push_back(k);
    return keys;
}
