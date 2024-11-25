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
                __func__, i, vars.rows, vars.cols);

        // multiply matrix
        matrix_dot(&dot_out, model.weights[i], vars);
#if 0
        log_debug("%s: after matrix_dot: ", __func__);
        for (int x = 0; x < dot_out.rows; ++x)
            for (int y = 0; y < dot_out.cols; ++y)
                log_debug("matrix[%d][%d] = %lf", x, y, matrix_get(dot_out, x, y));
#endif

        // add bias each term
        matrix_add_row(dot_out, model.biases[i]);
#if 0
        log_debug("%s: after matrix_add_row: ", __func__);
        for (int x = 0; x < dot_out.rows; ++x)
            for (int y = 0; y < dot_out.cols; ++y)
                log_debug("matrix[%d][%d] = %lf", x, y, matrix_get(dot_out, x, y));
#endif

        // Activation
        ai_activation_apply_matrix_column(dot_out, model.activation);
#if 0
        log_debug("%s: after ai_activation_apply_matrix_column: ", __func__);
        for (int x = 0; x < dot_out.rows; ++x)
            for (int y = 0; y < dot_out.cols; ++y)
                log_debug("matrix[%d][%d] = %lf", x, y, matrix_get(dot_out, x, y));
#endif

        // swap for later use.
        matrix_swap(&dot_out, &vars);
    }

    log_debug("%s: Model Predicted!", __func__);

    return vars;
}

void ai_model_train_gd(ai_model model, matrix data, matrix value)
{
    matrix dot_out = { 0 };
    matrix deriv = { 0 };
    matrix vars = matrix_copy(data);

    for (int i = 0; i < model.num_layers - 1; ++i)
    {
        // Y = Act(X * W + B)

        log_debug("%s: layer: %d, rows: %d, cols: %d",
                __func__, i, vars.rows, vars.cols);

        // multiply matrix. 
        matrix_dot(&dot_out, model.weights[i], vars);

        // add bias each term
        matrix_add_row(dot_out, model.biases[i]);
        
        // dot_out = z1

        // Activation
        ai_activation_apply_matrix_column(dot_out, model.activation);

        // dot_out = a1
        
        // dot_out - 
        
        // swap for later use.
        matrix_swap(&dot_out, &vars);
    }

    matrix out = vars;

    // Cost = (a-y)^2
    double cost = 0;
    for (int i = 0; i < out.cols; ++i)
    {
        double loss = 0;
        for (int j = 0; j < out.rows; ++j)
        {
            double v1 = matrix_get(out, j, i);
            double v2 = matrix_get(value, j, i);

            double sq = (v1 - v2) * (v1 - v2);

            loss += sq;
        }
        cost += loss;
    }
    cost /= (out.rows * out.cols);

    log_debug("%s: Cost: %lf", __func__, cost);
}
