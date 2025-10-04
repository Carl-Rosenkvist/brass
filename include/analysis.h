#ifndef ANALYSIS_H
#define ANALYSIS_H

#include <cmath>                 // std::round (used in MergeKey ctor)
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "binaryreader.h"
#include "histogram1d.h"
#include "datatree.h"
#include "mergekey.h"
class Analysis {
protected:
    MergeKeySet keys;
    std::string smash_version;

public:
    virtual ~Analysis() = default;
    virtual Analysis& operator+=(const Analysis& other) = 0;

    void set_merge_keys(MergeKeySet k);
    const MergeKeySet& get_merge_keys() const;

    void on_header(Header& header);

    const std::string& get_smash_version() const { return smash_version; }

    virtual void analyze_particle_block(const ParticleBlock& block, const Accessor& accessor) = 0;
    virtual void finalize() = 0;
    virtual void save(const std::string& save_dir_path) = 0;
    virtual void print_result_to(std::ostream& os) const {}
};

// ---------- Dispatcher ----------
class DispatchingAccessor : public Accessor {
public:
    void register_analysis(std::shared_ptr<Analysis> analysis);
    void on_particle_block(const ParticleBlock& block) override;
    void on_end_block(const EndBlock& block) override;
    void on_header(Header& header) override;

private:
    std::vector<std::shared_ptr<Analysis>> analyses;
}; 

// ---------- Result entry + run ----------
struct Entry {
    MergeKeySet key;
    std::shared_ptr<Analysis> analysis;
};

void run_analysis(const std::vector<std::pair<std::string, std::string>>& file_and_meta,
                  const std::string& analysis_name,
                  const std::vector<std::string>& quantities,
                  const std::string& output_folder = ".");

#endif // ANALYSIS_H
