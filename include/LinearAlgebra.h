#pragma once
#include <vector>

//A lousy approach to matrix multiplication xD
namespace LinearAlgebra{
    //multiplies matrix and vector
    std::vector <double> dot(std::vector<std::vector <double>>& matrix, std::vector <double>& vector){
        std::vector <double> result;
        for(auto& row: matrix){
            double weighted_sum = 0;
            for(double& element: row){
                int element_index = &element - &row[0];
                weighted_sum += element * vector[element_index];
            }
            result.push_back(weighted_sum);
        }
        return result;
    }

    //adds or subtracts two matrices
    std::vector <std::vector <double>> add_or_subtract(std::vector <std::vector <double>>& matrix1, std::vector <std::vector <double>>& matrix2, bool add = true){
        std::vector <std::vector <double>> result;
        result.resize(matrix1.size());
        for(auto& row: result){
            int row_index = &row - &result[0];
            row.resize(matrix1[row_index].size());
            for(double& element: row){
                int element_index = &element - &row[0];
                element = matrix1[row_index][element_index] + (add? 1 : -1) * matrix2[row_index][element_index];
            }
        }
        return result;
    }

    //adds or subtracts two vectors
    std::vector <double> add_or_subtract(std::vector <double>& vector1, std::vector <double>& vector2, bool add = true){
        std::vector <double> result;
        result.resize(vector1.size());
        for(double& element: result){
            int element_index = &element - &result[0];
            element = vector1[element_index] + (add? 1 : -1) * vector2[element_index];
        }
        return result;
    }

    //applies function to each element of matrix
    std::vector <std::vector <double>> apply_function(std::vector<std::vector <double>>& matrix, double (*function)(double)){
        std::vector <std::vector <double>> result;
        for(auto& row: matrix){
            std::vector <double> row_result;
            for(double& element: row){
                row_result.push_back((*function)(element));
            }
            result.push_back(row_result);
        }
        return result;
    }

    //applies function to each element of vector
    std::vector <double> apply_function(std::vector <double>& vector, double (*function)(double)){
        std::vector <double> result;
        for(auto& element: vector){
            result.push_back((*function)(element));
        }
        return result;
    }

    //multiplies matrix and scalar
    std::vector <std::vector <double>> multiply_by_scalar(std::vector<std::vector <double>>& matrix, double scalar){
        std::vector <std::vector <double>> result;
        for(auto& row: matrix){
            std::vector <double> row_result;
            for(double& element: row){
                row_result.push_back(scalar * element);
            }
            result.push_back(row_result);
        }
        return result;
    }

    //multiplies vector and scalar
    std::vector <double> multiply_by_scalar(std::vector <double>& vector, double scalar){
        std::vector <double> result;
        for(auto& element: vector){
            result.push_back(scalar * element);
        }
        return result;
    }

    //hadamard product of two vectors
    std::vector <double> hadamard_product(std::vector <double>& vector1, std::vector <double>& vector2){
        if(vector1.size() != vector2.size()) throw std::runtime_error("hadamard_product(): Size of vectors must be the same!");
        std::vector <double> result;
        result.resize(vector1.size());
        for(double& element: result){
            int element_index = &element - &result[0];
            element = vector1[element_index] * vector2[element_index];
        }
        return result;
    }

    //averages matrices of same size
    std::vector <std::vector <double>> find_average(std::vector <std::vector <std::vector <double>>>& matrices){
        std::vector <std::vector <double>> sum(matrices[0].size(), std::vector <double>(matrices[0][0].size()));
        for(auto& matrix: matrices){
            for(auto& row: matrix){
                int row_index = &row - &matrix[0];
                for(double& element: row){
                    int element_index = &element - &row[0];
                    sum[row_index][element_index] += element;
                }
            }
        }
        return multiply_by_scalar(sum, 1 / (double)matrices.size());
    }

    //averages vectors of same size
    std::vector <double> find_average(std::vector <std::vector <double>>& vectors){
        std::vector <double> sum(vectors[0].size());
        for(auto& vector: vectors){
            for(auto& element: vector){
                int element_index = &element - &vector[0];
                sum[element_index] += element;
            }
        }
        return multiply_by_scalar(sum, 1 / (double)vectors.size());
    }

    //transposes a matrix
    std::vector <std::vector <double>> transpose(std::vector <std::vector <double>>& matrix){
        std::vector <std::vector <double>> result;
        result.resize(matrix[0].size());
        for(auto& row: result){
            int row_index = &row - &result[0];
            row.resize(matrix.size());
            for(double& element: row){
                int column_index = &element - &row[0];
                element = matrix[column_index][row_index];
            }
        }
        return result;
    }
};