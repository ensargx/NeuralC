#include "matrix.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "util/logger.h"

static void matrix_ensure(matrix* pOut, int rows, int cols)
{
    if ( pOut->data )
    {
        if ( pOut->rows != rows || pOut->cols != cols )
        {
            matrix_destroy( pOut );
            matrix_init( pOut, rows, cols );
        }
    }
    else 
    {
        matrix_init(pOut, rows, cols);
    }
}

void matrix_init(matrix *pMatrix, int rows, int cols) 
{
    if (rows < 0 || cols < 0)
    {
        log_error("%s: Wrong parameters: rows || cols", __func__);
        return;
    }

    pMatrix->cols = cols;
    pMatrix->rows = rows;

    double **data = (double**) malloc(sizeof(double*) * rows);
    for(int i = 0; i < rows; ++i)
    {
        data[i] = malloc(cols * sizeof(double));
    }

    pMatrix->data = data;

    return;
}

void matrix_destroy(matrix *pMatrix)
{
    for (int i = 0; i < pMatrix->rows; ++i)
        free( pMatrix->data[i] );

    free( pMatrix->data );

    return;
}

void matrix_copy(matrix *pOut, matrix mat)
{
    matrix_ensure( pOut, mat.rows, mat.cols );

    for (int i = 0; i < mat.rows; ++i)
    {
        for (int j = 0; j < mat.cols; ++j)
        {
            matrix_set(*pOut, i, j, matrix_get(mat, i, j));
        }
    }
}  

double matrix_get(matrix matrix, int x, int y)
{
    if ( x >= matrix.rows || y >= matrix.cols )
    {
        log_error("%s: parameters out of bound. x: %d, rows: %d, y: %d, cols: %d.",
                __func__, x, matrix.rows, y, matrix.cols);
        return 0;
    }

    return matrix.data[x][y];
}

void matrix_set(matrix matrix, int x, int y, double value)
{
    if ( x >= matrix.rows || y >= matrix.cols )
    {
        log_error("%s: parameters out of bound. x: %d, rows: %d, y: %d, cols: %d.",
                __func__, x, matrix.rows, y, matrix.cols);
        return;
    }

    matrix.data[x][y] = value;

    return;
}

int matrix_dot(matrix* pOut, matrix x, matrix y)
{
    if ( x.cols != y.rows )
    {
        log_error("%s: Matrixes cannot be dot producted! x.cols: %d, y.rows: %d.",
                __func__, x.cols, y.rows);
        return 0;
    }

    matrix_ensure( pOut, x.rows, y.cols );

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
    double **data1 = pMat1->data;

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
                __func__, mat2.rows, mat2.cols, mat1.rows, mat1.cols);
        return;
    }

    for (int i = 0; i < mat1.rows; ++i)
    {
        for (int j = 0; j < mat1.cols; ++j)
        {
            double v1 = matrix_get(mat1, i, j);
            double v2 = matrix_get(mat2, i, 0);

            matrix_set(mat1, i, j, v1 + v2);
        }
    }

    return;
}

matrix matrix_transpose(matrix *pOut, matrix mat)
{
    if ( pOut->data != 0 )
        matrix_destroy( pOut );

    matrix_init( pOut, mat.cols, mat.rows );

    for (int i = 0; i < mat.cols; ++i)
        for (int j = 0; j < mat.rows; ++j)
            matrix_set(*pOut, i, j, matrix_get(mat, j, i));

    return *pOut;
}

void matrix_tanh(matrix* pOut, matrix mat)
{
    matrix_ensure( pOut, mat.rows, mat.cols );

    for (int i = 0; i < mat.rows; ++i)
    {
        for (int j = 0; j < mat.cols; ++j)
        {
            double val = matrix_get(mat, i, j);
            double new = tanh(val);
            matrix_set(*pOut, i, j, new);
        }
    }
}

double sigmoid(double val)
{
    return 1.0 / (1.0 + exp(-val));
}

void matrix_sigmoid(matrix mat)
{
    for (int i = 0; i < mat.rows; ++i)
    {
        for (int j = 0; j < mat.cols; ++j)
        {
            double val = matrix_get(mat, i, j);
            double new = sigmoid(val);
            matrix_set(mat, i, j, new);
        }
    }
}

matrix matrix_read_csv(const char* filename, int labeled)
{
    matrix mat = { 0 };

    // open file
    FILE* file = fopen(filename, "r");
    if ( file == 0 )
    {
        log_error("%s: Cannot open file %s!", __func__, filename);
        return mat;
    }

    int lineCount = 0;
    char c;
    while ((c = fgetc(file)) != EOF)
        if (c == '\n')
            ++lineCount;

    if ( labeled )
        --lineCount;

    fseek(file, 0, SEEK_SET);

    // read , count.
    int commaCount = 0;
    while ((c = fgetc(file)) != EOF)
    {
        if (c == ',')
            commaCount++;

        if (c == '\n')
            break;
    }

    log_debug( "%s: filename: %s, line count: %d, comma count: %d", __func__, filename, lineCount, commaCount );

    int rows = lineCount;
    int cols = commaCount + 1;

    fseek(file, 0, SEEK_SET);

    matrix_init(&mat, rows, cols);

    // Reading data line by line
    char line[4096] = { 0 };

    if (labeled)
    {
        char nl;
        while((nl = fgetc(file)) && (nl != '\n'))
            ;
    }

    int row = 0;
    while (fgets(line, sizeof(line), file) && row < rows)
    {
        char *token = strtok(line, ",");
        int col = 0;

        while (token != NULL && col < cols)
        {
            double val = strtod(token, NULL);
            matrix_set(mat, row, col, val);
            token = strtok(NULL, ",");
            col++;
        }
        row++;
    }

    fclose(file);

    return mat;


    for (int i = 0; i < cols; ++i)
    {
        for (int j = 0; j < rows; ++j)
        {
            // parse until ','
            double val = 0;
            (void)fscanf(file, "%lf", &val);
            matrix_set(mat, j, i, val);
            fgetc(file);
        }
    }

    return mat;
}

double sigmoid_deriv(double x)
{
    return sigmoid(x) * (1 - sigmoid(x));
}

void matrix_sigmoid_deriv(matrix *pOut, matrix mat)
{
    matrix_ensure( pOut, mat.rows, mat.cols );

    for (int i = 0; i < mat.rows; ++i)
    {
        for (int j = 0; j < mat.cols; ++j)
        {
            double val = matrix_get(mat, i, j);
            double deriv = sigmoid_deriv(val);
            matrix_set(*pOut, i, j, deriv);
        }
    }
}

double tanh_deriv(double x)
{
    return 1 - pow(tanh(x), 2);
}

void matrix_tanh_deriv(matrix *pOut, matrix mat)
{
    matrix_ensure( pOut, mat.rows, mat.cols );

    for (int i = 0; i < mat.rows; ++i)
    {
        for (int j = 0; j < mat.cols; ++j)
        {
            double val = matrix_get(mat, i, j);
            double deriv = tanh_deriv(val);
            matrix_set(*pOut, i, j, deriv);
        }
    }
}

void matrix_subtract(matrix* pOut, matrix a, matrix b)
{
    if ( a.rows != b.rows || a.cols != b.cols )
    {
        log_error("%s: matrix rows or cols did not match!", __func__);
        return;
    }

    if ( pOut->data )
        matrix_destroy( pOut );

    matrix_init(pOut, a.rows, a.cols);

    for (int i = 0; i < a.rows; ++i)
    {
        for (int j = 0; j < a.cols; ++j)
        {
            double val = matrix_get(a, i, j) - matrix_get(b, i, j);
            matrix_set(*pOut, i, j, val);
        }
    }
}

void matrix_scale(matrix *pOut, matrix mat, double val)
{
    matrix_ensure( pOut, mat.rows, mat.cols );

    for (int i = 0; i < mat.rows; ++i)
    {
        for (int j = 0; j < mat.cols; ++j)
        {
            matrix_set(*pOut, i, j, val * matrix_get(mat, i, j));
        }
    }
}

void matrix_mul(matrix *pOut, matrix mat, matrix sec)
{
    if ( mat.rows != sec.rows || mat.cols != sec.cols )
    {
        log_error("%s: matrixes does not have same dims!", __func__);
        return;
    }

    if ( pOut->data )
        matrix_destroy( pOut );

    matrix_init(pOut, mat.rows, mat.cols);

    for (int i = 0; i < mat.rows; ++i)
    {
        for (int j = 0; j < mat.cols; ++j)
        {
            double val = matrix_get(sec, i, j) * matrix_get(mat, i, j);
            matrix_set(*pOut, i, j, val);
        }
    }
}

void matrix_sum_rows(matrix* pOut, matrix m)
{
    if ( pOut->data )
        matrix_destroy( pOut );

    matrix_init(pOut, 1, m.cols);

    for (int j = 0; j < m.cols; ++j)
    {
        double sum = 0.0;
        for (int i = 0; i < m.rows; ++i)
        {
            sum += matrix_get(m, i, j);  // sum the values in the j-th column
        }
        matrix_set(*pOut, 0, j, sum);  // store the sum in the result matrix
    }

    return;
}

void matrix_add(matrix a, matrix b)
{
    if ( a.rows != b.rows || a.cols != b.cols )
    {
        log_error("%s: dimentions did not mnatch!", __func__);
        return;
    }

    for (int i = 0; i < a.rows; ++i)
    {
        for (int j = 0; j < a.cols; ++j)
        {
            double val = matrix_get(a, i, j) + matrix_get(b, i, j);
            matrix_set(a, i, j, val);
        }
    }
}

void matrix_zero(matrix mat)
{
    for(int i = 0; i < mat.rows; ++i)
    {
        for (int j = 0; j < mat.cols; ++j)
        {
            matrix_set(mat, i, j, 0);
        }
    }
}
