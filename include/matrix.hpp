#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <cstddef>

class Matrix {
public:
    Matrix(size_t num_row, size_t num_column);

    [[nodiscard]] float getElement(size_t row, size_t column) const;
    void setElement(size_t row, size_t column, float newValue);
    [[nodiscard]] size_t getRows() const;
    [[nodiscard]] size_t getColumns() const;
    float* getData();
    [[nodiscard]] const float* getData() const;


    [[nodiscard]] Matrix addMatrix(const Matrix& other) const;
    [[nodiscard]] Matrix rowSum() const;
    [[nodiscard]] Matrix addColumnVector(const Matrix &vector) const;
    [[nodiscard]] Matrix subtractMatrix(const Matrix& other) const;
    [[nodiscard]] Matrix multiplyScalar(float scalar) const;
    [[nodiscard]] Matrix multiplyElementWise(const Matrix &other) const;
    [[nodiscard]] Matrix multiplyMatrix(const Matrix& other) const;
    [[nodiscard]] Matrix transposeMatrix() const;

    [[nodiscard]] Matrix operator+(const Matrix &other) const;
    [[nodiscard]] Matrix operator-(const Matrix &other) const;
    [[nodiscard]] Matrix operator*(float scalar) const;
    [[nodiscard]] Matrix operator*(const Matrix &other) const;

private:
    void checkBounds(size_t row, size_t column) const;
    [[nodiscard]] size_t getIndex(size_t row, size_t column) const;

    size_t rows;
    size_t columns;
    std::vector<float> data;
};

#endif