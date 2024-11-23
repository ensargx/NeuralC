#include "activation.h"

#include <string.h>
#include <math.h>

static double ai_tanh(double x)
{
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

static double ai_tanh_deriv(double x)
{
    double tanh_x = ai_tanh(x);
    return 1.0 - tanh_x * tanh_x; 
}

ai_activation activation_tanh = {
    .name = "tanh",
    .calculate = ai_tanh,
    .derivetive = ai_tanh_deriv,
    .params = 0
};

static double ai_relu(double x)
{
    return (x > 0) ? x : 0.0;
}

static double ai_relu_deriv(double x)
{
    return (x > 0) ? 1.0 : 0.0;
}

ai_activation activation_relu = {
    .name = "ReLU",
    .calculate = ai_relu,
    .derivetive = ai_relu_deriv,
    .params = 0
};


#define NUM_ACTIVATIONS 2

ai_activation ai_activation_find(const char* name)
{
    ai_activation arr_activations[NUM_ACTIVATIONS] = {
        activation_tanh,
        activation_relu
    };

    for (int i = 0; i < NUM_ACTIVATIONS; ++i)
    {
        if ( strcmp(name, arr_activations[i].name) == 0 )
            return arr_activations[i];
    }

    return activation_tanh;
}

#undef NUM_ACTIVATIONS

void ai_activation_apply_matrix_column(matrix mat, ai_activation activation)
{
    for (int i = 0; i < mat.rows; ++i)
    {
        for (int j = 0; j < mat.cols; ++j)
        {
            double val = matrix_get(mat, i, j);
            double new = activation.calculate(val);
            matrix_set(mat, i, j, new);
        }
    }

    return;
}

