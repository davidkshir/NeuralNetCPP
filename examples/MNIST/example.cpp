#include "../../include/network.hpp"
#include "../../include/MNIST.hpp"


int main(){
    Network neural_net(0.01f, 784); // Network object creation

    // Layer creation, defaults to ReLU if no activation function is specified
    neural_net.addLayer(10, Activation::softmax);

    // Training set object creation
    const MNIST mnist_data(R"(\data\train-images-idx3-ubyte)", R"(data\train-labels-idx1-ubyte)");

    std::vector<Matrix> training_images;
    std::vector<Matrix> training_labels;

    mnist_data.loadImages(training_images);
    mnist_data.loadLabels(training_labels);

    neural_net.train(training_images, training_labels, 2, DebugLevel::Standard);
    neural_net.saveNetwork("Example1.nn");
}