#ifndef ANALYSIS_H
#define ANALYSIS_H

#include <yaml-cpp/yaml.h>

#include <cmath>  // std::round (used in MergeKey ctor)
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "binaryreader.h"
#include "datatree.h"
#include "histogram1d.h"
#include "mergekey.h"

class Analysis {
   protected:
    MergeKeySet keys;
    std::string smash_version;
    std::string analysis_name;

   public:
    explicit Analysis(const std::string &name) : analysis_name(name) {}
    virtual ~Analysis() = default;

    const std::string &name() const { return analysis_name; }
    const MergeKeySet &get_merge_keys() const { return keys; }
    void set_merge_keys(MergeKeySet k) { keys = std::move(k); }

    // Two analyses can combine iff they have the same analysis name AND same
    // merge keys
    virtual bool can_combine(const Analysis &other) const {
        if (analysis_name != other.analysis_name) return false;
        // If MergeKeySet has only operator<, equality = !(a<b) && !(b<a)
        const auto &a = keys;
        const auto &b = other.keys;
        return !(a < b) && !(b < a);
    }

    virtual Analysis &operator+=(const Analysis &other) = 0;

    void on_header(Header &header);
    const std::string &get_smash_version() const { return smash_version; }

    virtual void analyze_particle_block(const ParticleBlock &block,const Accessor &accessor) {};
    virtual void analyze_interaction_block(const InteractionBlock &block,const Accessor &accessor) {};
    virtual void analyze_end_block(const EndBlock &block,const Accessor &accessor) {};



    virtual void finalize() = 0;
    virtual void save(const std::string &save_dir_path) = 0;
    virtual void print_result_to(std::ostream &os) const {}
};
// ---------- Dispatcher ----------
class DispatchingAccessor : public Accessor {
   public:
    void register_analysis(std::shared_ptr<Analysis> analysis);
    void on_particle_block(const ParticleBlock &block) override;
    void on_interaction_block(const InteractionBlock &block) override;
    void on_end_block(const EndBlock &block) override;
    void on_header(Header &header) override;

   private:
    std::vector<std::shared_ptr<Analysis>> analyses;
};

// ---------- Result entry + run ----------
struct Entry {
    MergeKeySet key;
    std::shared_ptr<Analysis> analysis;
};

void run_analysis(
    const std::vector<std::pair<std::string, std::string>> &file_and_meta,
    const std::vector<std::string> &analysis_names,
    const std::vector<std::string> &quantities,
    const std::string &output_folder = ".");

#endif  // ANALYSIS_H
