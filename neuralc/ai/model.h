#pragma once

#include "activation.h"
#include "matrix/matrix.h"

/**
 * Every AI model should be respesentable in binary.
 * Also serializable in text and can be re-produced from a file.
 *
 * Each model will have `activation`, `layers`.
*/

typedef struct
{
    int num_layers;
    matrix* weights;
    matrix* biases;
    ai_activation activation;
} ai_model;
