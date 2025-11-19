#ifndef ANALYSIS_H
#define ANALYSIS_H

#include <yaml-cpp/yaml.h>

#include <memory>
#include <string>
#include <vector>

#include "binaryreader.h"
#include "blocks.h"
#include "mergekey.h"

class Analysis {
   protected:
    MergeKeySet keys;
    std::string smash_version;
    std::string analysis_name;

   public:
    explicit Analysis(const std::string& name) : analysis_name(name) {}
    virtual ~Analysis() = default;

    const std::string& name() const { return analysis_name; }
    const MergeKeySet& get_merge_keys() const { return keys; }
    void set_merge_keys(MergeKeySet k) { keys = std::move(k); }

    void on_header(Header& header);
    const std::string& get_smash_version() const { return smash_version; }

    virtual void analyze_particle_block(const ParticleBlock&, const Accessor&) {
    }
    virtual void analyze_interaction_block(const InteractionBlock&,
                                           const Accessor&) {}
    virtual void analyze_end_block(const EndBlock&, const Accessor&) {}

    virtual py::dict finalize(py::dict results) {
        return results;
    }

    virtual void save(py::dict results, const std::string& out_dir) {}

    virtual py::dict to_state_dict() const = 0;

    virtual void print_result_to(std::ostream& os) const {}
};
// ---------- Dispatcher ----------
class DispatchingAccessor : public Accessor {
   public:
    void register_analysis(std::shared_ptr<Analysis> analysis);
    void on_particle_block(const ParticleBlock& block) override;
    void on_interaction_block(const InteractionBlock& block) override;
    void on_end_block(const EndBlock& block) override;
    void on_header(Header& header) override;

    void create_and_register_analysis(const std::string& analysis_name);

   private:
    std::vector<std::shared_ptr<Analysis>> analyses;
};

#endif  // ANALYSIS_H
