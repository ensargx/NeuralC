#include "matrix.h"
#include <string.h>
#include <stdlib.h>

#include "util/logger.h"

void matrix_init(matrix *pMatrix, int rows, int cols) 
{
    if (rows < 0 || cols < 0)
    {
        log_error("[%s]: Wrong parameters: rows || cols", __FUNCTION__);
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
