#pragma once

typedef struct matrix_t
{
    int rows;
    int cols;
    double *data;
} matrix;

/**
 * Initializes the matrix with given rows and cols.
 *
 * Parameters:
 * rows: row count of the matrix.
 * cols: column count of the matrix.
*/
void matrix_init(matrix *pMatrix, int rows, int cols);

/**
 * Destroys the matrix and free's the memory. The matrix
 * will be unusable.
*/
void matrix_destroy(matrix *pMatrix);

/** 
 * Creates a copy of the given matrix.
*/
matrix matrix_copy(matrix* pMatrix);

/**
 * Gets the given element of the matrix.
 * Equalivent of `matrix->data[x * matrix->rows + y]`
 *
 * Parameters:
 * x: row number.
 * y: column number.
*/
double matrix_get(matrix *pMatrix, int x, int y);

/**
 * Set a value of the matrix.
 * Equal to `matrix->data[x * matrix->rows + y] = value`
*/
void matrix_set(matrix *pMatrix, int x, int y, double value);
