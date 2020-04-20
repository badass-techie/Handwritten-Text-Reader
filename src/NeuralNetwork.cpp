#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <utility>
#include <cstdlib>
#include <random>
#include <ctime>
#include <chrono>
#include <cmath>
#include "../include/NeuralNetwork.h"
#include "../include/LinearAlgebra.h"

NeuralNetwork::NeuralNetwork(std::vector<int> sizes, double min_random_weight, double max_random_weight): neurons_in_each_layer(sizes), no_of_layers(sizes.size() - 1){
    srand(time(0)); //seeds random seed
    for(int layer = 1; layer <= no_of_layers; ++layer){
        std::vector <std::vector <double>> weight_matrix;
        std::vector <double> bias_vector;
        for(int row = 0; row < sizes[layer]; ++row){
            std::vector <double> each_row_in_weight_matrix;
            for(int element = 0; element < sizes[layer - 1]; ++element){
                each_row_in_weight_matrix.push_back(NeuralNetwork::random(min_random_weight, max_random_weight));
            }
            weight_matrix.push_back(each_row_in_weight_matrix);
        }
        weights.push_back(weight_matrix);
        bias_vector.assign(sizes[layer], 0);
        biases.push_back(bias_vector);
    }
}

NeuralNetwork::NeuralNetwork(const char* file_to_read){
    read_from_file(file_to_read);
}

int NeuralNetwork::feed_forward(std::vector <double>& inputs){
    activation_inputs.clear();
    activations.clear();
    activations.push_back(inputs);

    for(auto& weight_matrix: weights){
        int matrix_index = &weight_matrix - &weights[0];

        //z = wa + b
        auto wa = LinearAlgebra::dot(weight_matrix, activations[matrix_index]);
        auto b = biases[matrix_index];
        auto z = LinearAlgebra::add_or_subtract(wa, b);
        activation_inputs.push_back(z);

        //a = σ(Z)
        auto a = LinearAlgebra::apply_function(z, NeuralNetwork::sigmoid);
        activations.push_back(a);
    }
    //returns the index of the maximum activation in the output layer
    return std::distance(activations.back().begin(), std::max_element(activations.back().begin(), activations.back().end()));
}

void NeuralNetwork::train(list_of_inputs_and_outputs& training_set, int epochs, int mini_batch_size, double eta, std::string file_to_write){
    for (int epoch = 1; epoch <= epochs; ++epoch){
        //std::shuffle(training_set.begin(), training_set.end(), std::default_random_engine(time(0)));    //shuffles training set
        //splits training set into mini batches
        int no_of_mini_batches = (training_set.size() - 1)/mini_batch_size + 1;
        for(int mini_batch_index = 0; mini_batch_index < no_of_mini_batches; ++mini_batch_index){
            //iterators for each mini batch
            auto start_itr = std::next(training_set.begin(), mini_batch_index * mini_batch_size);
            auto end_itr = std::next(training_set.begin(), mini_batch_index * mini_batch_size + mini_batch_size);

            //mini batch
            list_of_inputs_and_outputs mini_batch;
            mini_batch.resize(mini_batch_size);

            //resize last mini batch if it should contain less elements
            if (mini_batch_index * mini_batch_size + mini_batch_size > training_set.size()){
                end_itr = training_set.end();
                mini_batch.resize(training_set.size() - mini_batch_index * mini_batch_size);
            }

            //copy respective elements from training set to mini batch
            std::copy(start_itr, end_itr, mini_batch.begin());

            std::cout<<"Mini batch "<<mini_batch_index + 1<<" of "<<no_of_mini_batches<<". Cost: "<<stochastic_gradient_descent(mini_batch, eta)<<std::endl;
        }
        std::cout<<"Epoch "<<epoch<<" of "<<epochs<<" completed."<<std::endl;
    }
    if (file_to_write != "") write_to_file(file_to_write);
}

void NeuralNetwork::train(list_of_inputs_and_outputs& training_set, int epochs, int mini_batch_size, double eta, list_of_inputs_and_outputs& test_set, std::string file_to_write){
    int evaluation = 0;
    for (int epoch = 1; epoch <= epochs; ++epoch){
        costs.clear();
        cost_derivatives.clear();
        std::shuffle(training_set.begin(), training_set.end(), std::default_random_engine(time(0)));    //shuffles training set
        //splits training set into mini batches
        int no_of_mini_batches = (training_set.size() - 1)/mini_batch_size + 1;
        for(int mini_batch_index = 0; mini_batch_index < no_of_mini_batches; ++mini_batch_index){
            //iterators for each mini batch
            auto start_itr = std::next(training_set.begin(), mini_batch_index * mini_batch_size);
            auto end_itr = std::next(training_set.begin(), mini_batch_index * mini_batch_size + mini_batch_size);

            //mini batch
            list_of_inputs_and_outputs mini_batch;
            mini_batch.resize(mini_batch_size);

            //resize last mini batch if it should contain less elements
            if (mini_batch_index * mini_batch_size + mini_batch_size > training_set.size()){
                end_itr = training_set.end();
                mini_batch.resize(training_set.size() - mini_batch_index * mini_batch_size);
            }

            //copy respective elements from training set to mini batch
            std::copy(start_itr, end_itr, mini_batch.begin());

            std::cout<<"Mini batch "<<mini_batch_index + 1<<" of "<<no_of_mini_batches<<". Cost: "<<stochastic_gradient_descent(mini_batch, eta)<<std::endl;
        }
        evaluation = 0; //number of elements in test set predicted correctly after each epoch
        for(auto& test_sample: test_set) evaluation += test_sample.second[feed_forward(test_sample.first)] == 1.0;
        std::cout<<"Epoch "<<epoch<<" of "<<epochs<<" completed. Evaluation: "<<evaluation<<" / "<<test_set.size()<<std::endl;
    }
    if (file_to_write != "") write_to_file(file_to_write, 100 * evaluation/test_set.size());
}

double NeuralNetwork::stochastic_gradient_descent(list_of_inputs_and_outputs& mini_batch, double eta){
    std::vector <std::vector <std::vector <std::vector <double>>>> nabla_w_vector(no_of_layers);    //list of lists, with each list containing all matrices from nabla_w whose layer correspond to the index
    std::vector <std::vector <std::vector <double>>> nabla_b_vector(no_of_layers);    //list of lists, with each list containing all vectors from nabla_b whose layer correspond to the index
    for(auto& training_data: mini_batch){
        auto nabla_wb = back_propagation(training_data);
        for(auto& matrix: nabla_wb.first){
            int layer = &matrix - &nabla_wb.first[0];
            nabla_w_vector[layer].push_back(matrix);
        }
        for(auto& vector: nabla_wb.second){
            int layer = &vector - &nabla_wb.second[0];
            nabla_b_vector[layer].push_back(vector);
        }
    }
    
    double avg_cost = std::accumulate(costs.begin(), costs.end(), 0.0f)/costs.size();

    //average across the mini batch
    std::vector <std::vector <std::vector <double>>> avg_nabla_w;
    std::vector <std::vector <double>> avg_nabla_b;
    for(auto& list_of_matrices_in_layer: nabla_w_vector)
        avg_nabla_w.push_back(LinearAlgebra::find_average(list_of_matrices_in_layer));
    for(auto& list_of_vectors_in_layer: nabla_b_vector)
        avg_nabla_b.push_back(LinearAlgebra::find_average(list_of_vectors_in_layer));
    
    //adjust weights
    for(auto& matrix: avg_nabla_w){
        int layer = &matrix - &avg_nabla_w[0];
        auto delta_w = LinearAlgebra::multiply_by_scalar(matrix, eta);
        weights[layer] = LinearAlgebra::add_or_subtract(weights[layer], delta_w, false);
    }
    //adjust biases
    for(auto& vector: avg_nabla_b){
        int layer = &vector - &avg_nabla_b[0];
        auto delta_b = LinearAlgebra::multiply_by_scalar(vector, eta);
        biases[layer] = LinearAlgebra::add_or_subtract(biases[layer], delta_b, false);
    }

    return avg_cost;
}

nabla_w_nabla_b NeuralNetwork::back_propagation(input_and_output& i_o){
    std::vector <double> inputs = i_o.first, outputs = i_o.second;
    nabla_w_nabla_b result;
    feed_forward(inputs);

    //This neural network uses the quadratic cost function, (1/2n) * ∑||(A-Y)²||, to estimate the error in its output
    //The cost becomes (1/2) * ||(A-Y)²|| w.r.t. each training output
    //Since the output has many neurons, ||(A-Y)²|| = ∑(A-Y)², the sum of (A-Y)² over all the neurons
    double cost = 0;
    for (double& activation: activations.back()){
        int index = &activation - &activations.back()[0];
        cost += pow(activation - outputs[index], 2) / 2;
    }
    costs.push_back(cost);
    
    //Let δl be to what extent the cost affects the activation inputs in the last layer
    //δl is therefore a vector of ∂C/∂Z
    //From the chain rule, ∂C/∂Z = ∂C/∂A * ∂A/∂Z
    //C = (1/2) * (A-Y)²   ∂C/∂A = A-Y
    //A = σ(Z)      ∂A/∂Z = σ'(Z)
    //Thus δl = ∂C/∂Z = (A-Y) * σ'(Z)
    //The hadamard product is used, as these are two vectors
    std::vector <std::vector <double>> delta(no_of_layers);
    auto cost_derivative = LinearAlgebra::add_or_subtract(activations.back(), outputs, false);
    cost_derivatives.push_back(cost_derivative);
    auto avg_cost_derivative = LinearAlgebra::find_average(cost_derivatives);
    auto activation_derivative = LinearAlgebra::apply_function(activation_inputs.back(), NeuralNetwork::sigmoid_prime);
    delta.back() = LinearAlgebra::hadamard_product(cost_derivative, activation_derivative);

    //Let δl be δ for layer l and δ(l+1) be δ for the layer after l
    //δl = ∂C/∂Z(l) and δ(l+1) = ∂C/∂Z(l+1)
    //Applying the chain rule, ∂C/∂Z(l) = ∂C/∂Z(l+1) * ∂Z(l+1)/∂Z(l) which in turn = δ(l+1) * ∂Z(l+1)/∂Z(l)
    //Z(l+1) = W(l+1)·A(l) + b(l+1) = W(l+1)·σ(Z(l)) + b(l+1)   ∂Z(l+1)/∂Z(l) = W(l+1) * σ'(Z(l))
    //Thus δl = W(l+1)·δ(l+1) * σ'(Z(l))
    //For the dot product to work, the W(l+1) matrix has to be transposed to make its columns equal to the rows in δ(l+1)
    //δl is then the hadamard product of the resulting vector and σ'(Z(l))
    for (int index = no_of_layers - 2; index >= 0; --index){
        //starts from -2 to avoid the last element
        auto weight_matrix = LinearAlgebra::transpose(weights[index + 1]);
        auto dot_product = LinearAlgebra::dot(weight_matrix, delta[index+1]);
        auto activation_derivative = LinearAlgebra::apply_function(activation_inputs[index], NeuralNetwork::sigmoid_prime);
        delta[index] = LinearAlgebra::hadamard_product(dot_product, activation_derivative);
    }

    //Let δl be δ for layer l
    //δl = ∂C/∂Z(l)
    //From the chain rule, ∂C/∂B(l) = ∂C/∂Z(l) * ∂Z(l)/∂B(l)
    //Z(l) = W(l)A(l-1) + B(l)      ∂Z(l)/∂B(l) = 1
    //This proves that ∂C/∂B(l) = ∂C/∂Z(l) = δl
    result.second = delta;

    //Let δl be δ for layer l
    //δl = ∂C/∂Z(l)
    //From the chain rule, ∂C/∂W(l) = ∂C/∂Z(l) * ∂Z(l)/∂W(l)
    //Z(l) = W(l)A(l-1) + B(l)      ∂Z(l)/∂W(l) = A(l-1)
    //Thus ∂C/∂W(l) = A(l-1) * δl
    //This is done component wise for every weight
    for(auto& weight_matrix: weights){
        int matrix_index = &weight_matrix - &weights[0];
        std::vector <std::vector <double>> nabla_w_matrix(weight_matrix.size(), std::vector <double>(weight_matrix[0].size()));
        for (auto& row: weight_matrix){
            int row_index = &row - &weight_matrix[0];
            for (auto& element: row){
                int element_index = &element - &row[0];
                nabla_w_matrix[row_index][element_index] = activations[matrix_index][element_index] * delta[matrix_index][row_index];
            }
        }
        result.first.push_back(nabla_w_matrix);
    }
    
    return result;
}

void NeuralNetwork::read_from_file(std::string filename){
    std::ifstream file(filename);
    if (file.is_open()){
        std::vector <std::stringstream> lines;
        std::string token;
        while(std::getline(file, token)) lines.push_back((std::stringstream)token);
        int current_line = 6;

        //sizes
        while(std::getline(lines[current_line], token, ',')) neurons_in_each_layer.push_back(std::stoi(token));
        no_of_layers = neurons_in_each_layer.size() - 1;
        current_line+=3;

        //weights
        weights.resize(no_of_layers);
        for(int layer = 0; layer < no_of_layers; ++layer){
            weights[layer].resize(neurons_in_each_layer[layer+1]);
            for(int line = current_line; line < current_line+neurons_in_each_layer[layer+1]; ++line){
                while(std::getline(lines[line], token, ',')){
                    weights[layer][line - current_line].push_back(std::stod(token));
                }
            }
            current_line += neurons_in_each_layer[layer+1] + 2;
        }

        //biases
        biases.resize(no_of_layers);
        for(int layer = 0; layer < no_of_layers; ++layer){
            while(std::getline(lines[current_line], token, ',')){
                biases[layer].push_back(std::stod(token));
            }
            current_line += 3;
        }
        file.close();
    } else {
        throw std::runtime_error("Unable to read file");
    }
}

void NeuralNetwork::write_to_file(std::string filename, double success_rate){
    std::ofstream file(filename, std::ios::trunc);
    if(file.is_open()){
        //metadata
        //time
        time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::string time(30, '\0');
        std::strftime(&time[0], time.size(), "%H:%M:%S %d/%m/%y", std::localtime(&now));
        file<<"Created on:"<<','<<time<<'\n'<<'\n';
        //accuracy
        file<<"Accuracy:"<<','<<(success_rate < 0? "unknown" : (std::to_string(success_rate) + "%"))<<'\n'<<'\n';
        //size
        file<<"Number of neurons in: "<<'\n';
        for(int& size: neurons_in_each_layer)
            file<<((&size - &neurons_in_each_layer[0] == 0)? "Input Layer" : ((&size - &neurons_in_each_layer[0] == neurons_in_each_layer.size() - 1)? "Output Layer" : ("Hidden Layer "  + std::to_string(&size - &neurons_in_each_layer[0]))))<<',';
        file<<'\n';
        for(int& size: neurons_in_each_layer) file<<size<<',';
        file<<'\n'<<'\n';
        //weights
        for (auto& weight_matrix: weights){
            file<<((&weight_matrix - &weights[0] == weights.size() - 1)? "Weights in Output Layer" : "Weights in Hidden Layer " + std::to_string(&weight_matrix - &weights[0] + 1))<<'\n';
            for (auto& row: weight_matrix){
                for (double& weight: row) file<<weight<<',';
                file<<'\n';
            }
            file<<'\n';
        }
        //biases
        for (auto& bias_vector: biases){
            file<<((&bias_vector - &biases[0] == biases.size() - 1)? "Biases in Output Layer" : "Biases in Hidden Layer " + std::to_string(&bias_vector - &biases[0] + 1))<<'\n';
            for (double& bias: bias_vector) file<<bias<<',';
            file<<'\n'<<'\n';
        }
        file.close();
    } else{
        throw std::runtime_error("Failed to write to file");
    }
}

inline double NeuralNetwork::random(double min, double max){
    return min + static_cast<double>(rand())/static_cast<double>(RAND_MAX/(max - min));
}

inline double NeuralNetwork::sigmoid(double z){
    return 1/(1 + std::exp(-z));
}

inline double NeuralNetwork::sigmoid_prime(double z){
    return NeuralNetwork::sigmoid(z) * (1 - NeuralNetwork::sigmoid(z));
}