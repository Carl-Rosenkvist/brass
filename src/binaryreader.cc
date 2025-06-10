#include "binaryreader.h"


const std::unordered_map<std::string, QuantityInfo> quantity_string_map = {
    {"t",            {Quantity::T,             QuantityType::Double}},
    {"x",            {Quantity::X,             QuantityType::Double}},
    {"y",            {Quantity::Y,             QuantityType::Double}},
    {"z",            {Quantity::Z,             QuantityType::Double}},
    {"mass",         {Quantity::MASS,          QuantityType::Double}},
    {"p0",           {Quantity::P0,            QuantityType::Double}},
    {"px",           {Quantity::PX,            QuantityType::Double}},
    {"py",           {Quantity::PY,            QuantityType::Double}},
    {"pz",           {Quantity::PZ,            QuantityType::Double}},
    {"pdg",          {Quantity::PDG,           QuantityType::Int32}},
    {"id",           {Quantity::ID,            QuantityType::Int32}},       
    {"charge",       {Quantity::CHARGE,        QuantityType::Int32}},
    {"ncoll",        {Quantity::NCOLL,         QuantityType::Int32}},
    {"form_time",    {Quantity::FORM_TIME,     QuantityType::Double}},
    {"xsecfac",      {Quantity::XSEC_FACTOR,  QuantityType::Double}},
    {"proc_id_origin",   {Quantity::PROC_ID_ORIGIN,   QuantityType::Int32}},
    {"proc_type_origin", {Quantity::PROC_TYPE_ORIGIN, QuantityType::Int32}},
    {"t_last_coll",  {Quantity::TIME_LAST_COLL, QuantityType::Double}},
    {"pdg_mother1",  {Quantity::PDG_MOTHER1,   QuantityType::Int32}},
    {"pdg_mother2",  {Quantity::PDG_MOTHER2,   QuantityType::Int32}},
    {"baryon_number",{Quantity::BARYON_NUMBER, QuantityType::Int32}},
    {"strangeness",  {Quantity::STRANGENESS,   QuantityType::Int32}},
};



size_t type_size(QuantityType t) {
    switch (t) {
        case QuantityType::Double: return sizeof(double);
        case QuantityType::Int32:  return sizeof(int32_t);
        default: throw std::logic_error("Unknown QuantityType");
    }
}

std::unordered_map<Quantity, size_t>
compute_quantity_layout(const std::vector<std::string>& names) {
    std::unordered_map<Quantity, size_t> layout;
    size_t offset = 0;

    for (const auto& name : names) {
        auto it = quantity_string_map.find(name);
        if (it == quantity_string_map.end())
            throw std::runtime_error("Unknown quantity: " + name);

        layout[it->second.quantity] = offset;
        offset += type_size(it->second.type); 
    }

    return layout;
}

std::vector<char> read_chunk(std::ifstream& bfile, size_t size) {
    std::vector<char> buffer(size);
    bfile.read(buffer.data(), size);
    if (!bfile) throw std::runtime_error("Read failed");
    return buffer;
}

void Accessor::set_layout(const std::unordered_map<Quantity, size_t>* layout_in) {
    layout = layout_in;
}

int32_t Accessor::get_int(const std::string& name, const ParticleBlock& block, size_t i) const {
    return quantity<int32_t>(name, block, i);
}

double Accessor::get_double(const std::string& name, const ParticleBlock& block, size_t i) const {
    return quantity<double>(name, block, i);
}

BinaryReader::BinaryReader(const std::string& filename,
                           const std::vector<std::string>& selected,
                           std::shared_ptr<Accessor> accessor_in)
    : file(filename, std::ios::binary), accessor(std::move(accessor_in))
{
    if (!file) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    layout = compute_quantity_layout(selected);

    for (const std::string& name : selected) {
        const auto& info = quantity_string_map.at(name);
        particle_size += type_size(info.type);
    }

    if (!accessor) throw std::runtime_error("An accessor is needed!");
    accessor->set_layout(&layout);
}

void BinaryReader::read() {
    Header header = Header::read_from(file);
    char blockType;
    if(accessor) accessor->on_header(header);
    while (file.read(&blockType, sizeof(blockType))) {
        switch (blockType) {
            case 'p': {
                ParticleBlock p_block = ParticleBlock::read_from(file, particle_size);
                if (accessor && check_next(file)) accessor->on_particle_block(p_block);
                break;
            }
            case 'f': {
                EndBlock e_block = EndBlock::read_from(file);
                if (accessor && check_next(file)) accessor->on_end_block(e_block);
                break;
            }
            case 'i':
                break;
            default:
                break;
        }
    }
}


bool BinaryReader::check_next(std::ifstream& bfile) {
    int c = bfile.peek();
    if (c == EOF) return false;
    char t = static_cast<char>(c);
    return t == 'p' || t == 'f' || t == 'i';
}

