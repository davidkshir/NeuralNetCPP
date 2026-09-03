#include "network.hpp"
#include "mnist.hpp"


int main(){

    Network neural_net(0.01f, 784);

    // Layer creation, defaults to ReLU if no activation function is specified
    neural_net.addLayer(128);
    neural_net.addLayer(64);
    neural_net.addLayer(10, Activation::softmax);

    // Load the MNIST training data and labels
    const mnist training_data(R"(data\train-images-idx3-ubyte)", R"(data\train-labels-idx1-ubyte)");

    // Create vectors of matrices to store training samples and labels
    std::vector<Matrix> training_images;
    std::vector<Matrix> training_labels;

    training_data.loadImages(training_images);
    training_data.loadLabels(training_labels);

    // Train for 3 epochs with standard progress output
    neural_net.train(training_images, training_labels, 3, DebugLevel::Standard);


    neural_net.saveNetwork("examples/MNIST/Example.nn");

    Network network = Network::loadNetwork("examples/MNIST/Example.nn");

    const mnist testing_data(R"(test\t10k-images-idx3-ubyte)", R"(test\t10k-labels-idx1-ubyte)");

    std::vector<Matrix> testing_images;
    std::vector<Matrix> testing_labels;

    testing_data.loadImages(testing_images);
    testing_data.loadLabels(testing_labels);

    size_t correct = 0;

    for (size_t i = 0; i < testing_images.size(); ++i) {
        const Matrix output = network.inference(testing_images[i]);

        size_t predicted = 0;
        size_t expected = 0;

        // Find the predicted and expected digit
        for (size_t j = 1; j < output.getRows(); ++j) {
            if (output.getElement(j, 0) > output.getElement(predicted, 0)) {
                predicted = j;
            }

            if (testing_labels[i].getElement(j, 0) > testing_labels[i].getElement(expected, 0)) {
                expected = j;
            }
        }

        if (predicted == expected) {
            correct++;
        }
    }

    const double accuracy = static_cast<double>(correct) / static_cast<double>(testing_images.size()) * 100.0;

    std::cout << "Test Accuracy: " << accuracy << "%\n";
}