#ifndef LAYER_H
#define LAYER_H

#include "matrix.hpp"

enum class Activation {
    reLU,
    softmax
};

class Layer {
    public:
        Layer(size_t inputs, size_t neurons, Activation function = Activation::reLU); // Use ReLU as default activation function

        Matrix forward(Matrix const &input);

        Matrix back(Matrix const &outputGradient);

        void update(float learning_rate);

        [[nodiscard]] size_t getNeurons() const;

        [[nodiscard]] Matrix& getWeights();

        [[nodiscard]] const Matrix& getWeights() const;

        [[nodiscard]] Matrix& getBiases();

        [[nodiscard]] const Matrix& getBiases() const;
        
        [[nodiscard]] Activation getActFunction() const;

    private:
        static Matrix reLU(Matrix const &input);

        static Matrix softmax(Matrix const &input);

        static Matrix reLUDerivative(Matrix const &input);
       
        Matrix weights;
        Matrix biases;
        Matrix inputCache; // Stores layers input
        Matrix preActCache; // Stores neurons values before activation function
        Activation function;
        Matrix weight_gradient;
        Matrix bias_gradient;
};

#endif