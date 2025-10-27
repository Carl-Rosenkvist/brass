#ifndef BINARY_READER_H
#define BINARY_READER_H

#include "accessor.h"
#include "quantities.h"

#include <memory>


class BinaryReader {
   public:
    BinaryReader(const std::string &filename,
                 const std::vector<std::string> &selected,
                 std::shared_ptr<Accessor> accessor_in);
    void read();

   private:
    std::ifstream file;
    size_t particle_size = 0;
    std::shared_ptr<Accessor> accessor;
    std::unordered_map<std::string, size_t> layout;

};

#endif  // BINARY_READER_H
