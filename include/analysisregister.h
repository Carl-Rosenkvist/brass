#pragma once
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "analysis.h"

class AnalysisRegistry {
public:
    // Zero-arg factory
    using Factory = std::function<std::shared_ptr<Analysis>()>;

    static AnalysisRegistry& instance();

    void register_factory(const std::string& name, Factory factory);
    std::shared_ptr<Analysis> create(const std::string& name) const; // 'name' is the key

    std::vector<std::string> list_registered() const;

private:
    std::unordered_map<std::string, Factory> factories_;
};

// Registers CLASS under key NAME. The lambda captures NAME and constructs CLASS(NAME).
#define REGISTER_ANALYSIS(NAME, CLASS)                           \
    static bool _registered_##CLASS = []() {                     \
        AnalysisRegistry::instance().register_factory(           \
            NAME, []() -> std::shared_ptr<Analysis> {            \
                return std::make_shared<CLASS>(NAME);            \
            });                                                  \
        return true;                                             \
    }();                                                         \
    static const void* _anchor_##CLASS = &_registered_##CLASS;
