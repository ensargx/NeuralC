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

        // add bias each term
        matrix_add_row(dot_out, model.biases[i]);

        // Activation
        ai_activation_apply_matrix_column(dot_out, model.activation);

        // swap for later use.
        matrix_swap(&dot_out, &vars);
    }

    log_debug("%s: Model Predicted!", __FUNCTION__);

    return vars;
}
