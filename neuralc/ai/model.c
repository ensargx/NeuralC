#include "model.h"
#include "matrix/matrix.h"
#include "util/logger.h"

matrix ai_model_predict(ai_model model, matrix data)
{
    matrix dot_out = { 0 };
    matrix vars = matrix_copy(data);

    for (int i = 0; i < model.num_layers - 1; ++i)
    {
        // Y = Act(X * W + B)

        log_debug("%s: layer: %d, rows: %d, cols: %d",
                __FUNCTION__, i, vars.rows, vars.cols);

        // multiply matrix
        matrix_dot(&dot_out, model.weights[i], vars);
        log_debug("%s: after matrix_dot: ", __FUNCTION__);
        for (int x = 0; x < dot_out.rows; ++x)
            for (int y = 0; y < dot_out.cols; ++y)
                log_debug("matrix[%d][%d] = %lf", x, y, matrix_get(dot_out, x, y));

        // add bias each term
        matrix_add_row(dot_out, model.biases[i]);
        log_debug("%s: after matrix_add_row: ", __FUNCTION__);
        for (int x = 0; x < dot_out.rows; ++x)
            for (int y = 0; y < dot_out.cols; ++y)
                log_debug("matrix[%d][%d] = %lf", x, y, matrix_get(dot_out, x, y));

        // Activation
        ai_activation_apply_matrix_column(dot_out, model.activation);
        log_debug("%s: after ai_activation_apply_matrix_column: ", __FUNCTION__);
        for (int x = 0; x < dot_out.rows; ++x)
            for (int y = 0; y < dot_out.cols; ++y)
                log_debug("matrix[%d][%d] = %lf", x, y, matrix_get(dot_out, x, y));

        // swap for later use.
        matrix_swap(&dot_out, &vars);
    }

    log_debug("%s: Model Predicted!", __FUNCTION__);

    return vars;
}
