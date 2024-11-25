#include "matrix.h"
#include <string.h>
#include <stdlib.h>

#include "util/logger.h"

void matrix_init(matrix *pMatrix, int rows, int cols) 
{
    if (rows < 0 || cols < 0)
    {
        log_error("%s: Wrong parameters: rows || cols", __FUNCTION__);
        return;
    }

    pMatrix->cols = cols;
    pMatrix->rows = rows;

    double *data = (double*) malloc(sizeof(double) * rows * cols);
    pMatrix->data = data;

    return;
}

void matrix_destroy(matrix *pMatrix)
{
    if ( pMatrix->data )
        free( pMatrix->data );

    return;
}

matrix matrix_copy(matrix mat)
{
    matrix res;

    matrix_init(&res, mat.rows, mat.cols);

    memcpy(res.data, mat.data, mat.rows * mat.cols * sizeof(double));

    return res;
}  

double matrix_get(matrix matrix, int x, int y)
{
    if ( x >= matrix.rows || y >= matrix.cols )
    {
        log_error("%s: parameters out of bound. x: %d, rows: %d, y: %d, cols: %d.",
                __FUNCTION__, x, matrix.rows, y, matrix.cols);
        return 0;
    }

    return matrix.data[ x * matrix.cols + y ];
}

void matrix_set(matrix matrix, int x, int y, double value)
{
    if ( x >= matrix.rows || y >= matrix.cols )
    {
        log_error("%s: parameters out of bound. x: %d, rows: %d, y: %d, cols: %d.",
                __FUNCTION__, x, matrix.rows, y, matrix.cols);
        return;
    }

    matrix.data[ x * matrix.cols + y ] = value;

    return;
}

int matrix_dot(matrix* pOut, matrix x, matrix y)
{
    if ( x.cols != y.rows )
    {
        log_error("%s: Matrixes cannot be dot producted! x.cols: %d, y.rows: %d.",
                __FUNCTION__, x.cols, y.rows);
        return 0;
    }

    // destroy pOut if not null
    if ( pOut->data != 0 )
        free( pOut->data );

    matrix_init(pOut, x.rows, y.cols);

    for (int i = 0; i < pOut->rows; ++i)
    {
        for (int j = 0; j < pOut->cols; ++j)
        {
            double val = 0;
            for (int h = 0; h < x.cols; ++h)
            {
                val += matrix_get(x, i, h) * matrix_get(y, h, j);
            }
            matrix_set(*pOut, i, j, val);
        }
    }

    return 1;
}

matrix matrix_create_random(int rows, int cols, double lower, double upper, int seed)
{
    matrix mat;

    matrix_init(&mat, rows, cols);

    srand(seed);

    double range = upper - lower;
    double div = RAND_MAX / range;

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            double val = lower + (rand() / div);
            matrix_set(mat, i, j, val);
        }
    }

    return mat;
}

void matrix_swap(matrix *pMat1, matrix *pMat2)
{
    int rows1 = pMat1->rows;
    int cols1 = pMat1->cols;
    double *data1 = pMat1->data;

    pMat1->rows = pMat2->rows;
    pMat1->cols = pMat2->cols;
    pMat1->data = pMat2->data;

    pMat2->rows = rows1;
    pMat2->cols = cols1;
    pMat2->data = data1;
}

void matrix_add_row(matrix mat1, matrix mat2)
{
    if ( mat2.cols != 1 || mat2.rows != mat1.rows )
    {
        log_error("%s: Matrixes not aligned properly! mat2.rows: %d, mat2.cols: %d, mat1.rows: %d, mat1.cols: %d",
                __FUNCTION__, mat2.rows, mat2.cols, mat1.rows, mat1.cols);
        return;
    }

    for (int i = 0; i < mat1.cols ; i++)
    {
        for (int j = 0; j < mat2.rows ; ++j)
        {
            double v1 = matrix_get(mat1, j, i);
            double v2 = matrix_get(mat2, j, 0);

            matrix_set(mat1, j, i, v1 + v2);
        }
    }

    return;
}
