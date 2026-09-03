#include "network.hpp"
#include <filesystem>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <cmath>

Network::Network(const float learning_rate, const size_t input_size)
    : input_size(input_size),
      learning_rate(learning_rate)
{
    if(learning_rate <= 0){
        throw std::invalid_argument("Learning rate must be > 0.");
    }

    if(input_size <= 0){
        throw std::invalid_argument("Input size must be > 0.");
    }
}

        void Network::addLayer(size_t neurons, Activation function) {
            if(layers.empty()){
                layers.emplace_back(input_size, neurons, function);
            }
            else{
                layers.emplace_back(layers.back().getNeurons(), neurons, function);
            }
        }

        Matrix Network::forwardPropagation(Matrix const &input) {
            if (layers.empty()){
                throw std::invalid_argument("Network cannot have 0 layers.");
            }

            Matrix prev_layer_output = input;

            for(Layer &layer : layers){
                prev_layer_output = layer.forward(prev_layer_output);
            }

            return prev_layer_output;
        }

        float Network::crossEntropyLoss(Matrix const &prediction, Matrix const &target) {
            if(prediction.getRows() != target.getRows() || prediction.getColumns() != target.getColumns()){
                throw std::invalid_argument("Matrices must have same dimensions for loss calculation.");
            }

            if(prediction.getColumns() == 0){
                throw std::invalid_argument("Cannot calculate loss for empty batch.");
            }

            float loss = 0.0f;

            float safe_predict = 0.0f;
            for(size_t i = 0; i < prediction.getRows(); i++){
                for(size_t j = 0; j < prediction.getColumns(); j++){
                    safe_predict = std::max(1e-7f, prediction.getElement(i, j)); // Restricts smallest output to 1e-7f for log() math

                    loss += target.getElement(i, j) * std::log(safe_predict);
                }
            }
            
            loss = -loss;
            return loss;
        }

        Matrix Network::crossEntropyGradient(Matrix const &prediction, Matrix const &target) {
            if(prediction.getRows() != target.getRows() || prediction.getColumns() != target.getColumns()){
                throw std::invalid_argument("Matrices must have same dimensions for loss calculation.");
            }

            if(prediction.getColumns() == 0){
                throw std::invalid_argument("Cannot calculate loss for empty batch.");
            }

            return prediction - target; // Softmax + cross-entropy derivative simplifies to (prediction - target)
        }

        Matrix Network::backPropagation(Matrix const &outputGradient) {
            if (layers.empty()){
                throw std::invalid_argument("Network cannot have 0 layers.");
            }

            Matrix current_gradient = outputGradient;

            for(size_t i = layers.size() - 1; ; i--){

                current_gradient = layers[i].back(current_gradient);

                if (i == 0){
                    break;
                }
            }

            return current_gradient;
        }

        void Network::updateNetwork(){
            for(Layer &layer: layers ){
                layer.update(learning_rate);
            }
        }

        void Network::saveNetwork(std::filesystem::path const &path) const {
            std::ofstream save_file(path, std::ios::binary);
            if(!save_file){
                throw std::runtime_error("Failed to open/crete network save file.");
            }

            const auto layer_count = static_cast<uint32_t>(layers.size());

            // Save file header construction
            save_file.write(reinterpret_cast<const char*>(&MAGIC_NUMBER), sizeof(MAGIC_NUMBER));
            save_file.write(reinterpret_cast<const char*>(&VERSION), sizeof(VERSION));
            save_file.write(reinterpret_cast<const char*>(&learning_rate), sizeof(learning_rate));
            save_file.write(reinterpret_cast<const char*>(&input_size), sizeof(input_size));
            save_file.write(reinterpret_cast<const char*>(&layer_count), sizeof(layer_count));

            for(size_t i = 0; i < layer_count; i++){

                auto layer_neuron_count = static_cast<uint32_t>(layers[i].getNeurons());
                Activation layer_act_function = layers[i].getActFunction();

                // Store weight & bias matrices to avoid making extra redundant copies
                Matrix layer_weights = layers[i].getWeights();
                Matrix layer_biases = layers[i].getBiases();

                const float* p_layer_weights = layer_weights.getData();
                const float* p_layer_biases = layer_biases.getData();

                // Save file body construction
                save_file.write(reinterpret_cast<const char*>(&layer_neuron_count), sizeof(layer_neuron_count));
                save_file.write(reinterpret_cast<const char*>(&layer_act_function), sizeof(layer_act_function));
                save_file.write(reinterpret_cast<const char*>(p_layer_weights),  static_cast<std::streamsize>(sizeof(float) * (layer_weights.getRows() * layer_weights.getColumns())));
                save_file.write(reinterpret_cast<const char*>(p_layer_biases), static_cast<std::streamsize>(sizeof(float) * (layer_biases.getRows() * layer_biases.getColumns())));
            }
        }

        Network Network::loadNetwork(std::filesystem::path const &path) {
            std::ifstream load_file(path, std::ios::binary);
            if(!load_file){
                throw std::runtime_error("Failed to read network save file.");
            }

            uint32_t magic_number;
            uint32_t version;
            float learning_rate;
            uint32_t input_size;
            uint32_t layer_count;


            // Reads header
            load_file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
            load_file.read(reinterpret_cast<char*>(&version), sizeof(version));
            load_file.read(reinterpret_cast<char*>(&learning_rate), sizeof(learning_rate));
            load_file.read(reinterpret_cast<char*>(&input_size), sizeof(input_size));
            load_file.read(reinterpret_cast<char*>(&layer_count), sizeof(layer_count));

            if(magic_number != MAGIC_NUMBER){
                throw std::invalid_argument("Incorrect filetype.");
            }

            if(version != VERSION){
                throw std::invalid_argument("Incorrect version.");
            }

            Network network(learning_rate, input_size);

            uint32_t prev_size = input_size; // Prev size starts as network input_size

            for(size_t i = 0; i < layer_count; i++){
                uint32_t layer_neuron_count;
                Activation layer_act_function;

                load_file.read(reinterpret_cast<char*>(&layer_neuron_count), sizeof(layer_neuron_count));
                load_file.read(reinterpret_cast<char*>(&layer_act_function), sizeof(layer_act_function));

                network.addLayer(layer_neuron_count, layer_act_function);

                uint32_t weights_bytes_size = layer_neuron_count * prev_size * sizeof(float);
                load_file.read(reinterpret_cast<char*>(network.layers[i].getWeights().getData()), weights_bytes_size);

                if(!load_file){
                    throw std::runtime_error("Failed to read network save file.");
                }

                uint32_t biases_bytes_size = layer_neuron_count * sizeof(float);
                load_file.read(reinterpret_cast<char*>(network.layers[i].getBiases().getData()), biases_bytes_size);

                if(!load_file){
                    throw std::runtime_error("Failed to read network save file.");
                }

                prev_size = network.layers[i].getNeurons();
            }

            return network;
        }

        void Network::train(const std::vector<Matrix>& training_data, const std::vector<Matrix>& training_labels, const size_t epochs, const DebugLevel debug_level ) {
            if(training_data.empty()){
                throw std::invalid_argument("Cannot have empty dataset.");
            }

            if(training_labels.empty()){
                throw std::invalid_argument("Cannot have empty label set.");
            }

            if(training_data.size() != training_labels.size()){
                throw std::invalid_argument("Dataset size must match label set.");
            }

            for(const Matrix &data: training_data){
                if(data.getRows() != input_size){
                    throw std::invalid_argument("Dataset size must match input size");
                }

                if(data.getColumns() != 1){
                    throw std::invalid_argument("Data should be column vectors.");
                }
            }

            if (layers.empty()) {
                throw std::invalid_argument("Network must have at least one layer.");
            }

            const size_t output_size = layers.back().getNeurons();

            for(const Matrix &label: training_labels){
                if(label.getRows() != output_size){
                    throw std::invalid_argument("Label data size must match output size");
                }

                if(label.getColumns() != 1){
                    throw std::invalid_argument("Labels should be column vectors.");
                }
            }

            Matrix network_output(training_labels[0].getRows(), 1);
            Matrix output_grad(training_labels[0].getRows(), 1);
            float loss = 0.0f;
            float avg_loss = 0.0f;
            size_t correct_predictions = 0;
            const float* output_max_element = nullptr;
            const float* label_max_element = nullptr;
            size_t output_element_index = 0;
            size_t label_element_index = 0;
            size_t total_processed = 0;

            if(debug_level != DebugLevel::None)
                std::cout
                        << "============================================" << '\n'
                        << "Training Started" << '\n'
                        << "============================================" << '\n' << '\n'
                        << "Dataset Size: " << training_data.size() << '\n'
                        << "Epochs: " << epochs << '\n'
                        << "Input Size: " << input_size << '\n'
                        << "Output Size: " << output_size << '\n'
                        << "Learning Rate: " << learning_rate << '\n' << '\n';

            for(size_t e = 0; e < epochs; e++){ // Outer loop controls # of epochs

                for(size_t i = 0; i < training_data.size(); i++) {
                    total_processed = e * training_data.size() + (i + 1);


                    // Use references to avoid copying data every iteration
                    const Matrix& current_input = training_data[i];
                    const Matrix& current_label = training_labels[i];

                    network_output = forwardPropagation(current_input);

                    loss = crossEntropyLoss(network_output, current_label);
                    avg_loss += loss; // Still needs to be divided by i

                    output_max_element = std::max_element(network_output.getData(), network_output.getData() + network_output.getRows() * network_output.getColumns());
                    output_element_index = output_max_element - network_output.getData();

                    label_max_element = std::max_element(training_labels[i].getData(), training_labels[i].getData() + training_labels[i].getRows() * training_labels[i].getColumns());
                    label_element_index = label_max_element - training_labels[i].getData();

                    if(output_element_index == label_element_index){
                        correct_predictions++;
                    }

                    if((i + 1) % 100 == 0 && debug_level == DebugLevel::Standard){
                        std::cout
                                << "\033[2K" << "--------------------------------------------" << '\n'
                                << "\033[2K" << "Epoch: " << e + 1 << " / " << epochs << '\n'
                                << "\033[2K" << "Progress: " << i + 1 << " / " << training_data.size() << ' ' << "(" << (static_cast<float>(i) + 1.0f) /  static_cast<float>(training_data.size()) * 100.0f << "%)" << '\n' << '\n'

                                << "\033[2K" << "Average Loss: " << avg_loss / static_cast<float>(total_processed) << '\n'
                                << "\033[2K" << "Accuracy: " <<  static_cast<float>(correct_predictions) /  static_cast<float>(total_processed) * 100.0f << '\n'

                                << "\033[2K" << "--------------------------------------------" << '\n' << "\033[7A";
                    }

                    output_grad = crossEntropyGradient(network_output, current_label);

                    backPropagation(output_grad);

                    updateNetwork();
                }
                if(debug_level == DebugLevel::Basic){
                    std::cout
                            << "\033[2K" << "--------------------------------------------" << '\n'
                            << "\033[2K" << "Epoch: " << e + 1 << " / " << epochs << '\n'
                            << "\033[2K" << "Average Loss: " << avg_loss / static_cast<float>(total_processed) << '\n'
                            << "\033[2K" << "Accuracy: " << static_cast<float>(correct_predictions) / (static_cast<float>(total_processed) + 1.0f) * 100.0f << '\n'
                            << "\033[2K" << "--------------------------------------------" << '\n';
                }
            }
            std::cout
                        << "\033[7B" << '\n'
                        << "============================================" << '\n'
                        << "Training Completed" << '\n'
                        << "============================================" << '\n';

        }

        Matrix Network::inference(const Matrix& input) {
            return forwardPropagation(input);
        }

        