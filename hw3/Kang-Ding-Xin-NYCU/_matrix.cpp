#include <iostream>
#include <iomanip>
#include <vector>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <mkl.h>

class Matrix {

public:

    Matrix(size_t nrow, size_t ncol)
      : m_nrow(nrow), m_ncol(ncol)
    {
        reset_buffer(nrow, ncol);
    }

    Matrix(Matrix const & other)
      : m_nrow(other.m_nrow), m_ncol(other.m_ncol)
    {
        reset_buffer(other.m_nrow, other.m_ncol);
        for (size_t i=0; i<m_nrow; ++i)
        {
            for (size_t j=0; j<m_ncol; ++j)
            {
                (*this)(i,j) = other(i,j);
            }
        }
    }

    Matrix & operator=(Matrix const & other)
    {
        if (this == &other) { return *this; }
        if (m_nrow != other.m_nrow || m_ncol != other.m_ncol)
        {
            reset_buffer(other.m_nrow, other.m_ncol);
        }
        for (size_t i=0; i<m_nrow; ++i)
        {
            for (size_t j=0; j<m_ncol; ++j)
            {
                (*this)(i,j) = other(i,j);
            }
        }
        return *this;
    }

    // TODO: move constructors and assignment operators.

    ~Matrix()
    {
        reset_buffer(0, 0);
    }

    double   operator() (size_t row, size_t col) const
    {
        return m_buffer[index(row, col)];
    }
    double & operator() (size_t row, size_t col)
    {
        return m_buffer[index(row, col)];
    }

    size_t nrow() const { return m_nrow; }
    size_t ncol() const { return m_ncol; }

    size_t size() const { return m_nrow * m_ncol; }
    double buffer(size_t i) const { return m_buffer[i]; }
    std::vector<double> buffer_vector() const
    {
        return std::vector<double>(m_buffer, m_buffer+size());
    }

    bool is_transposed() const { return m_transpose; }

    Matrix & transpose()
    {
        m_transpose = !m_transpose;
        std::swap(m_nrow, m_ncol);
        return *this;
    }

private:

    size_t index(size_t row, size_t col) const
    {
        if (m_transpose) { return row          + col * m_nrow; }
        else             { return row * m_ncol + col         ; }
    }

    void reset_buffer(size_t nrow, size_t ncol)
    {
        if (m_buffer) { delete[] m_buffer; }
        const size_t nelement = nrow * ncol;
        if (nelement) { m_buffer = new double[nelement]; }
        else          { m_buffer = nullptr; }
        m_nrow = nrow;
        m_ncol = ncol;
    }

    size_t m_nrow = 0;
    size_t m_ncol = 0;
    bool m_transpose = false;
    double * m_buffer = nullptr;
};

Matrix multiply_naive(Matrix const & m1, Matrix const & m2){
    if(m1.ncol()!=m2.nrow()){
        throw std::invalid_argument("wrong dimention");
    }
    Matrix res(m1.nrow(), m2.ncol());
    for(int i = 0; i < m1.nrow; ++i){
        for(int j = 0; j < m2.ncol(); ++j){
            double element = 0;
            for(int k = 0; k < m1.ncol; ++k){
                element += m1(i, k)*m2(k, j);
            }
            res(i, j) = element;
        }
    }
    return res;
}

Matrix multiply_tile(Matrix const & m1, Matrix const & m2){
    if(m1.ncol()!=m2.nrow()){
        throw std::invalid_argument("wrong dimention");
    }
    Matrix res(m1.nrow(), m2.ncol());
    for(int i = 0; i < m1.nrow; ++i){
        for(int j = 0; j < m2.ncol(); ++j){
            double element = 0;
            for(int k = 0; k < m1.ncol; ++k){
                element += m1(i, k)*m2(k, j);
            }
            res(i, j) = element;
        }
    }
    return res;
}

Matrix multiply_mkl(Matrix const & m1, Matrix const & m2){
    if(m1.ncol()!=m2.nrow()){
        throw std::invalid_argument("wrong dimention");
    }
    Matrix res(m1.nrow(), m2.ncol());
    m = m1.nrow();
    n = m2.ncol();
    k = m1.ncol();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, 1.0, m1, k, m2, n, 1.0, res, n);
    return res;
}