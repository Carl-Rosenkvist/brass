#include "binaryreader.h"

BinaryReader::BinaryReader(const std::string &filename,
                           const std::vector<std::string> &selected,
                           std::shared_ptr<Accessor> accessor_in)
    : file(filename, std::ios::binary), accessor(std::move(accessor_in)) {
    if (!file) throw std::runtime_error("Could not open file: " + filename);

    layout = compute_quantity_layout(selected);

    for (const std::string &name : selected) {
        const auto &type = quantity_string_map.at(name);
        particle_size += type_size(type);
    }

    if (!accessor) throw std::runtime_error("An accessor is needed!");
    accessor->set_layout(&layout);
    accessor->set_resolved_fields(selected);
}




void BinaryReader::read() {
    Header hdr = Header::read_from(file);
    if (accessor) accessor->on_header(hdr);

    Format fmt{}; // or: Format fmt = hdr.as_format();

    char blockType;
    while (file.read(&blockType, sizeof(blockType))) {
        switch (blockType) {
            case 'p': {
                auto p_block = ParticleBlock::read_from(file, particle_size, fmt);
                if (accessor) accessor->on_particle_block(p_block);
                break;
            }
            case 'f': {
                auto e_block = EndBlock::read_from(file, fmt);
                if (accessor) accessor->on_end_block(e_block);
                break;
            }
            case 'i': {
                auto i_block = InteractionBlock::read_from(file, particle_size, fmt);
                if (accessor) accessor->on_interaction_block(i_block);
                break;
            }
            default:
                // unknown tag — bail out gracefully
                return;
        }
    }
}
