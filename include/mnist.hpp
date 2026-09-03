#ifndef MNIST_H
#define MNIST_H

#include <vector>
#include <filesystem>
#include <cstdint>
#include "matrix.hpp"

class mnist {
    public:
        mnist(std::filesystem::path const &data, std::filesystem::path const &label);

        void loadImages(std::vector<Matrix>& training_images) const;

        void loadLabels(std::vector<Matrix>& training_labels) const;

        [[nodiscard]] uint32_t getImageCount() const;

    private:
        static uint32_t endianConversion(unsigned char const *bytes);

        struct Header{
            uint32_t magic_number{};
            uint32_t image_count{};
            uint32_t rows{};
            uint32_t columns{};
        };

        Header header;
        std::vector<unsigned char> pixels;
        std::vector<unsigned char> labels;
};


#endif