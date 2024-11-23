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

ai_activation ai_activation_find(const char* name)
{
    if ( strcmp(name, activation_relu.name) == 0 )
        return activation_relu;

}
