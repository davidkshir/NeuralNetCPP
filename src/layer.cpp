#include "layer.hpp"
#include <cmath>
#include <random>
#include <stdexcept>

Layer::Layer(const size_t inputs, const size_t neurons, const Activation function) // Use ReLU as default activation function
        : weights(neurons, inputs),
        biases(neurons, 1),
        inputCache(inputs, 1),
        preActCache(neurons, 1),
        function(function),
        weight_gradient(neurons, inputs),
        bias_gradient(neurons, 1)
        {

            const float standardDeviation = std::sqrt(2.0f / static_cast<float>(inputs));
            std::random_device rd;
            std::mt19937 engine(rd());
            std::normal_distribution distribution(0.0f, standardDeviation);

            for(size_t i = 0; i < neurons; i++){
                for(size_t j = 0; j < inputs; j++){
                    weights.setElement(i, j, distribution(engine));
                }
            }
        }

Matrix Layer::forward(Matrix const &input) {
            inputCache = input; // Saves layers input for backpropagation
            preActCache = (weights * input) + biases; // Saves pre-activation values for backpropagation

            switch(function){
                case Activation::reLU:
                    return reLU(preActCache);

                case Activation::softmax:
                    return softmax(preActCache);

                default:
                    throw std::invalid_argument("Unsupported activation function.");
            }
        }

Matrix Layer::back(Matrix const &outputGradient) {
            Matrix activation_gradient(outputGradient.getRows(), outputGradient.getColumns());

            switch(function){ 
                case Activation::reLU:
                    activation_gradient = outputGradient.multiplyElementWise(reLUDerivative(preActCache));
                    break;

                case Activation::softmax:
                    activation_gradient = outputGradient;
                    break;

                default:
                    throw std::invalid_argument("Unsupported activation function.");
            }

            weight_gradient = activation_gradient * inputCache.transposeMatrix(); 

            bias_gradient = activation_gradient;

            Matrix input_gradient = weights.transposeMatrix() * activation_gradient;

            return input_gradient;
        }

void Layer::update(float learning_rate) {
            weights = weights - (weight_gradient * learning_rate);
            biases = biases - (bias_gradient * learning_rate);
        }
size_t Layer::getNeurons() const {
            return weights.getRows();
        }

Matrix& Layer::getWeights() {
            return weights;
        }
const Matrix& Layer::getWeights() const {
            return weights;
        }
Matrix& Layer::getBiases() {
            return biases;
        }

const Matrix& Layer::getBiases() const {
            return biases;
        }

Activation Layer::getActFunction() const {
            return function;
        }

 Matrix Layer::reLU(Matrix const &input) {
            Matrix output(input.getRows(), input.getColumns());

            for(size_t i = 0; i < input.getRows(); i++){
                for(size_t j = 0; j < input.getColumns(); j++){
                    if(input.getElement(i, j) < 0){
                        output.setElement(i, j, 0);
                        continue;
                    }

                    output.setElement(i, j, input.getElement(i, j));
                }
            }
            return output;
        }

Matrix Layer::softmax(Matrix const &input) {
            Matrix output(input.getRows(), input.getColumns());

            for (size_t i = 0; i < input.getColumns(); i++) {

                float exponentialSum = 0.0f;
                float max = input.getElement(0, i); // Starts as first value in Matrix in case of negative floats

                
                for(size_t j = 0; j < input.getRows(); j++){
                    if(input.getElement(j, i) > max){
                        max = input.getElement(j, i);
                    }
                }

                for(size_t j = 0; j < input.getRows(); j++){
                    exponentialSum += std::exp(input.getElement(j, i) - max);
                }
                
            
                for(size_t j = 0; j < input.getRows(); j++){
                    output.setElement(j, i, (std::exp(input.getElement(j, i) - max) / exponentialSum));
                }
            }

            return output;
        }
       
Matrix Layer::reLUDerivative(Matrix const &input) {
            Matrix output(input.getRows(), input.getColumns());

            for(size_t i = 0; i < input.getRows(); i++){
                for(size_t j = 0; j < input.getColumns(); j++){
                    if(input.getElement(i, j) <= 0){
                        output.setElement(i, j, 0.0f);
                        continue;
                    }

                    output.setElement(i, j, 1.0f);
                }
            }
            return output;
        }