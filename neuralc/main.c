#include "util/logger.h"
#include "matrix/matrix.h"

int main()
{
    log_debug("Testing...");

    matrix mat;

    matrix_init(&mat, 5, 10);

    matrix_set(mat, 1, 10, 3.14);
    
    log_debug("value: %lf", matrix_get(mat, 1, 10));


    matrix dot = {0};

    matrix testA;
    matrix_init(&testA, 2, 4);

    matrix testB;
    matrix_init(&testB, 4, 3);

    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            matrix_set(testA, i, j, i + j);
        }
    }

    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            matrix_set(testB, i, j, i + j);
        }
    }

    matrix_dot(&dot, testA, testB);

    for (int i = 0; i < dot.rows; ++i)
    {
        for (int j = 0; j < dot.cols; ++j)
        {
            double val = matrix_get(dot, i, j);
            log_debug("matrix[%d][%d] = %lf", i, j, val);
        }
    }


    matrix w1 = matrix_create_random(3, 5, -1, 1, 54);

    for (int i = 0; i < w1.rows; ++i)
    {
        for (int j = 0; j < w1.cols; ++j)
        {
            log_debug("w1[%d][%d] = %lf", i, j, matrix_get(w1, i, j));
        }
    }

    return 0;
}
