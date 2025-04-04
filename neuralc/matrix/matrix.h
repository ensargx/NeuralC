#pragma once

typedef struct
{
    int rows;
    int cols;
    double **data;
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
void matrix_copy(matrix *pOut, matrix matrix);

/**
 * Gets the given element of the matrix.
 * Equivalent to `matrix.data[x * matrix.rows + y];`
 *
 * Parameters:
 * x: row number.
 * y: column number.
*/
double matrix_get(matrix matrix, int x, int y);

/**
 * Set a value of the matrix.
 * Equivalent to `matrix->data[x * matrix->rows + y] = value;`
*/
void matrix_set(matrix matrix, int x, int y, double value);

/**
 * Gives the dot product of two matrix.
 * gives `X*Y`.
 * if pOut is not empty (matrix.data != null), it will destroy
 * the matrix first.
 * You should pass pOut after `matrix->data = null;` in 
 * first creation of matrix, or initialize matrix with 
 * `matrix dot = { 0 };`
 *
 * Parameters:
 * pOut: Pointer to matrix result will be written.
 * x: First matrix in the product.
 * y: Second matrix in the product.
 *
 * Returns:
 * 1 if succeeds, else 0.
*/
int matrix_dot(matrix* pOut, matrix x, matrix y);

/**
 * Creates a matrix with random variables with given seed.
 * Returns the same matrix if the seed is same.
*/
void matrix_create_random(matrix *pOut, int rows, int cols, double lower, double upper, int seed);

/**
 * Swaps 2 matrixes
*/
void matrix_swap(matrix *pMat1, matrix *pMat2);

/**
 * Adds mat2 for each row of mat1.
 * It is expected that `mat2.cols == 1` and 
 * `mat2.rows == mat1.rows`.
 * This will iterate every row and add corresponding
 * values.
*/
void matrix_add_row(matrix mat1, matrix mat2);

/**
 * Creates the transpose of a given matrix.
 * Initialize the pout matrix with `matrix trans = { 0 };`.
 * Also call `matrix_destroy( &out );` to free memory.
 *
 * Parameters:
 * - pOut: Output matrix, if data is not empty,
 *   it'll be destroyed.
 * - matrix: matrix to take its transpose.
 *
 * Returns:
 * pOut matrix.
*/
matrix matrix_transpose(matrix *pOut, matrix mat);

/**
 * Apply tanh function for every element in a matrix.
*/
void matrix_tanh(matrix* pOut, matrix mat);

/**
 * Apply sigmoid function for every element in matrix.
*/
void matrix_sigmoid(matrix mat);
void matrix_sigmoid_deriv(matrix *pOut, matrix mat);

/**
 * Read a csv file and translate it to matrix.
 *
 * Parameters:
 * - filename: file path of the csv.
 * - labeled: if the data is labeled, 1 else 0.
*/
matrix matrix_read_csv(const char* filename, int labeled);

void matrix_tanh_deriv(matrix *pOut, matrix mat);

/**
 * a - b
*/
void matrix_subtract(matrix* pOut, matrix a, matrix b);

/**
 * mat * val
*/
void matrix_scale(matrix *pOut, matrix mat, double val);

/**
 * pOut = mat * sec
*/
void matrix_mul(matrix *pOut, matrix mat, matrix sec);

/*
 * sums over matrix m in rows.
*/
void matrix_sum_rows(matrix* pOut, matrix m);

/*
 * a += b
*/
void matrix_add(matrix a, matrix b);
    
void matrix_zero(matrix mat);
