#include <vector>
#include <fstream>
#include <filesystem>
#include <cstdint>
#include <stdexcept>
#include "matrix.hpp"
#include "mnist.hpp"

mnist::mnist(std::filesystem::path const &data, std::filesystem::path const &label) {

            std::ifstream data_file(data, std::ios::binary);
            if (!data_file) {
                throw std::runtime_error("Failed to open image data file.");
            }

            unsigned char data_header[16];
            data_file.read(reinterpret_cast<char*>(data_header), 16);
            if (!data_file) {
                throw std::runtime_error("Failed to read image data's header.");
            }


            header.magic_number =  endianConversion(&data_header[0]);
            header.image_count = endianConversion(&data_header[4]);
            header.rows = endianConversion(&data_header[8]);
            header.columns = endianConversion(&data_header[12]);

            if(header.magic_number != 2051){
                throw std::invalid_argument("File is not .idx3 type.");
            }
            
            pixels.resize(header.image_count * header.rows * header.columns); // Allocate space before read() writes directly into the vector's memory
            data_file.read(reinterpret_cast<char*>(pixels.data()), header.rows * header.columns * header.image_count);
            if (!data_file) {
                throw std::runtime_error("Failed to read image data's pixels.");
            }

            std::ifstream labels_file(label, std::ios::binary);
            if (!labels_file) {
                throw std::runtime_error("Failed to open labels file.");
            }

            unsigned char labels_header[8];
            labels_file.read(reinterpret_cast<char*>(labels_header), 8);
            if (!labels_file) {
                throw std::runtime_error("Failed to read label's header.");
            }

            // Only useful for initial validation, so no need to store
            uint32_t labels_magic_number = endianConversion(&labels_header[0]);
            uint32_t labels_size = endianConversion(&labels_header[4]);

            if(labels_magic_number != 2049){
                throw std::invalid_argument("File is not .idx1 type.");
            }

            if(header.image_count != labels_size){
                throw std::invalid_argument("There must be a label for every set of training data.");
            }

            labels.resize(labels_size); // Allocate space before read() writes directly into the vector's memory
            labels_file.read(reinterpret_cast<char*>(labels.data()), labels_size);
            if (!labels_file) {
                throw std::runtime_error("Failed to read label's data.");
            }
        }

        void mnist::loadImages(std::vector<Matrix>& training_images) const {
            training_images.clear();
            training_images.reserve(header.image_count); // Avoids potential reallocation during emplace_back() avoiding invalid references & performance losses
            size_t image_size = header.rows * header.columns;

            for (size_t i = 0; i < header.image_count; i++){
                training_images.emplace_back(image_size, 1);
                Matrix& current_image = training_images.back();
                float* image_data = current_image.getData();

                for (size_t j = 0; j < image_size; j++){
                    image_data[j] = static_cast<float>(pixels[i * image_size + j]) / 255.0f;
                }
            }
        }

        void mnist::loadLabels(std::vector<Matrix>& training_labels) const {
            training_labels.clear();
            training_labels.reserve(header.image_count);
            
            for (size_t i = 0; i < header.image_count; i++){
                training_labels.emplace_back(10, 1);
                Matrix& current_label = training_labels.back();
                float* label_data = current_label.getData();
                label_data[labels[i]] = 1.0f;
            }
        }

        uint32_t mnist::getImageCount() const {
            return header.image_count;
        }

        uint32_t mnist::endianConversion(unsigned char const *bytes){
            // Constructs a uint32_t from four bytes stored in big-endian order.
            return (static_cast<uint32_t>(bytes[0]) << 24) |
            (static_cast<uint32_t>(bytes[1]) << 16) |
            (static_cast<uint32_t>(bytes[2]) << 8)  |
            static_cast<uint32_t>(bytes[3]);
        }