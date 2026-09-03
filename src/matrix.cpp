#include <stdexcept>
#include "matrix.hpp"

Matrix::Matrix(const size_t num_row, const size_t num_column) // Avoids creating multiple std::vectors by filling on initialization
        : rows(num_row), 
        columns(num_column), 
        data(std::vector(num_row * num_column, 0.0f))
        {
        }

float Matrix::getElement(const size_t row, const size_t column) const {

            return data[getIndex(row, column)];
        }

void Matrix::setElement(const size_t row, const size_t column, const float newValue) {

            data[getIndex(row, column)] = newValue;
        }

size_t Matrix::getRows() const {
            return rows;
        }

size_t Matrix::getColumns() const {
            return columns;
        }

float* Matrix::getData() {
            return data.data();
        }

const float* Matrix::getData() const {
            return data.data();
        }

Matrix Matrix::addMatrix(const Matrix &other) const {
            if(rows != other.rows || columns != other.columns){
                throw std::invalid_argument("Matrices must have same dimensions for addition.");
            }

            const size_t matrix_dim =  rows * columns;

            Matrix result(rows, columns); // Doesn't matter which matrix dimension is used because they're equal

            for (size_t i = 0; i < matrix_dim; i++){
                result.data[i] = data[i] + other.data[i];
            }

            return result;
        }
Matrix Matrix::rowSum() const {
            Matrix total_sum(rows, 1);
            float sum = 0.0f;

            for(size_t i = 0; i < rows; i++){
                for(size_t j = 0; j < columns; j++){
                    sum += data[getIndex(i, j)];
                }
                total_sum.setElement(i, 0, sum);
                sum = 0.0f;
            }
            return total_sum;
        }

Matrix Matrix::addColumnVector(const Matrix &vector) const {
            if(vector.getColumns() != 1 || vector.getRows() != rows){
                throw std::invalid_argument("Improper vector dimensions.");
            }

            Matrix output(rows, columns);

            for(size_t i = 0; i < columns; i++){
                for(size_t j = 0; j < rows; j++){
                    output.setElement(j, i, getElement(j, i) + vector.getElement(j, 0));
                }
            }
            return output;
        }

Matrix Matrix::subtractMatrix(const Matrix &other) const {
            if(rows != other.rows || columns != other.columns){
                throw std::invalid_argument("Matrices must have same dimensions for subtraction.");
            }
            const size_t matrix_dim =  rows * columns;

            Matrix result(rows, columns); // Doesn't matter which matrix dimension is used because they're equal

            for (size_t i = 0; i < matrix_dim; i++){
                result.data[i] = data[i] - other.data[i];
            }
            
            return result;
        } 

Matrix Matrix::multiplyScalar(const float scalar) const {

            const size_t matrix_dim =  rows * columns;

            Matrix result(rows, columns); 

            for(size_t i = 0; i <matrix_dim; i++){
                result.data[i] = data[i] * scalar;
            }

            return result;
        }

Matrix Matrix::multiplyElementWise(const Matrix &other) const {
            if(rows != other.rows || columns != other.columns){
                throw std::invalid_argument("Matrices must have same dimensions for element-wise multiplication.");
            }
            const size_t matrix_dim =  rows * columns;

            Matrix result(rows, columns); // Doesn't matter which matrix dimension is used because they're equal

            for (size_t i = 0; i < matrix_dim; i++){
                result.data[i] = data[i] * other.data[i];
            }
            
            return result;
        }

Matrix Matrix::multiplyMatrix(const Matrix &other) const {
            if (columns != other.rows){
                throw std::invalid_argument("Matrices have incompatible dimensions for multiplication.");
            }

            Matrix result(rows, other.columns); // Final matrix will have # of rows from 1st matrix and # of columns from 2nd matrix

            // Loops ordered to maximize cache hits (builds entire row at once progressively instead of completely building one memory block then moving to the next)
            for (size_t i = 0; i < rows; i++){
                for (size_t k = 0; k < columns; k ++){
                    for (size_t j = 0; j < other.columns; j++){
                        result.data[result.getIndex(i, j)] +=  data[getIndex(i, k)] * other.data[other.getIndex(k, j)];
                    }

                }
            }
            return result;
        }
        
Matrix Matrix::transposeMatrix() const {
            Matrix result(columns, rows); 

            for (size_t i = 0; i < rows; i++){
                for (size_t j = 0; j < columns; j++){
                    result.setElement(j, i, getElement(i, j));
                }
            }
            return result;
        }       

// Private method
void Matrix::checkBounds(const size_t row, const size_t column) const {
            if (row >= rows || column >= columns) {
                throw std::out_of_range("Matrix index out of bounds.");
            }
        }

size_t Matrix::getIndex(const size_t row, const size_t column) const {
            checkBounds(row, column);
            
            return columns * row + column;
        }

// Operator overloads for easier usage
Matrix Matrix::operator+(const Matrix &other) const {
            if (rows == other.rows && columns == other.columns){
                return addMatrix(other);
            }
            if(rows == other.rows && other.columns == 1){
                return addColumnVector(other);
            }

            throw std::invalid_argument("Matrices have incompatible dimensions.");
        }

Matrix Matrix::operator-(const Matrix &other) const {
            return subtractMatrix(other);
        }

Matrix Matrix::operator*(const float scalar) const {
            return multiplyScalar(scalar);
        }

Matrix Matrix::operator*(const Matrix &other) const {
            return multiplyMatrix(other);
        }
