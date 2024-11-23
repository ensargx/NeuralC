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

matrix matrix_copy(matrix* pMatrix)
{
    matrix mat;

    matrix_init(&mat, pMatrix->rows, pMatrix->cols);

    memcpy(&mat.data, pMatrix->data, pMatrix->rows * pMatrix->cols * sizeof(double));

    return mat;
}  

double matrix_get(matrix matrix, int x, int y)
{
    if ( x > matrix.rows || y > matrix.cols )
    {
        log_error("%s: parameters out of bound. x: %d, rows: %d, y: %d, cols: %d.",
                __FUNCTION__, x, matrix.rows, y, matrix.cols);
        return 0;
    }

    return matrix.data[ x * matrix.cols + y ];
}

void matrix_set(matrix matrix, int x, int y, double value)
{
    if ( x > matrix.rows || y > matrix.cols )
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

