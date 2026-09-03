#ifndef NETWORK_H
#define NETWORK_H

#include <filesystem>
#include <vector>
#include <cstddef>
#include <cstdint>
#include "layer.hpp"
#include "matrix.hpp"


enum class DebugLevel{
    None,
    Basic,
    Standard,
};

class Network{
    static constexpr uint32_t MAGIC_NUMBER = 0x4B4E4554; // "KNET", file identifier (My last initial and NET for network)
    static constexpr uint32_t VERSION = 1;

    public:
        Network(float learning_rate, size_t input_size);

        void addLayer(size_t neurons, Activation function = Activation::reLU);

        void saveNetwork(std::filesystem::path const &path) const;

        static Network loadNetwork(std::filesystem::path const &path);

        void train(const std::vector<Matrix>& training_data, const std::vector<Matrix>& training_labels, size_t epochs, DebugLevel debug_level = DebugLevel::Standard);

        Matrix inference(const Matrix& input);

    private:
        Matrix forwardPropagation(Matrix const &input);

        static float crossEntropyLoss(Matrix const &prediction, Matrix const &target);

        static Matrix crossEntropyGradient(Matrix const &prediction, Matrix const &target);

        Matrix backPropagation(Matrix const &outputGradient);

        void updateNetwork();

        uint32_t input_size;
        float learning_rate;
        std::vector<Layer> layers;
};

#endif