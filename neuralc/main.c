#include "ai/activation.h"
#include "ai/model.h"
#include "matrix/matrix.h"
#include "util/logger.h"
#include <stdlib.h>

int main()
{
    log_debug("Testing...");

    ai_model model;

    model.num_layers = 2;
    model.activation = activation_tanh;
    
    matrix *matw = malloc(sizeof(matrix));
    matrix_init(matw, 3, 5);

    matrix_set(*matw, 0, 0, 0.2);
    matrix_set(*matw, 1, 0, -0.5);
    matrix_set(*matw, 2, 0, 0.7);
    matrix_set(*matw, 0, 1, -0.1);
    matrix_set(*matw, 1, 1, 0.3);
    matrix_set(*matw, 2, 1, -0.4);
    matrix_set(*matw, 0, 2, 0.4);
    matrix_set(*matw, 1, 2, -0.2);
    matrix_set(*matw, 2, 2, 0.1);
    matrix_set(*matw, 0, 3, 0.5);
    matrix_set(*matw, 1, 3, 0.1);
    matrix_set(*matw, 2, 3, -0.6);
    matrix_set(*matw, 0, 4, -0.3);
    matrix_set(*matw, 1, 4, 0.6);
    matrix_set(*matw, 2, 4, 0.2);

    model.weights = matw;

    matrix* matb = malloc(sizeof(matrix));
    matrix_init(matb, 3, 1);

    matrix_set(*matb, 0, 0, 0.1);
    matrix_set(*matb, 1, 0, -0.2);
    matrix_set(*matb, 2, 0, 0.3);

    model.biases = matb;

    matrix matx;

    matrix_init(&matx, 5, 2);
    matrix_set(matx, 0, 0, 0.5);
    matrix_set(matx, 1, 0, 0.2);
    matrix_set(matx, 2, 0, -0.7);
    matrix_set(matx, 3, 0, 0.9);
    matrix_set(matx, 4, 0, -0.4);

    matrix_set(matx, 0, 1, 0.1);
    matrix_set(matx, 1, 1, 0.2);
    matrix_set(matx, 2, 1, 0);
    matrix_set(matx, 3, 1, 0.9);
    matrix_set(matx, 4, 1, -0.4);

    log_debug("Predicting model.");
    matrix y = ai_model_predict(model, matx);

    log_debug("result matrix matrix: rows: %d, cols: %d",y.rows, y.cols);

    for (int i = 0; i < y.rows; ++i)
    {
        for (int j = 0; j < y.cols; ++j)
        {
            log_debug("matrix[%d][%d] = %lf", i, j, matrix_get(y, i, j));
        }
    }

    return 0;
}
