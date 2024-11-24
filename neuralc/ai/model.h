#pragma once

#include "activation.h"
#include "matrix/matrix.h"

/**
 * Every AI model should be respesentable in binary.
 * Also serializable in text and can be re-produced from a file.
*/

typedef struct
{
    int num_layers;
    matrix* weights;
    matrix* biases;
    ai_activation activation;
} ai_model;

/**
 * Predicts data from given model and data.
 * Returns a matrix with rows of `data->rows` and 
 * cols of `model->layers[num_layers-1].num_nodes`.
*/
matrix ai_model_predict(ai_model model, matrix data);




