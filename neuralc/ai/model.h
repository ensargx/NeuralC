#pragma once

#include "layer.h"
#include "activation.h"
#include "matrix/matrix.h"

/**
 * Every AI model should be respesentable in binary.
 * Also serializable in text and can be re-produced from a file.
*/

typedef struct
{
    int num_layers;
    ai_layer *layers;
    matrix* weights;
    matrix* biases;
    ai_activation activation;
} ai_model;




