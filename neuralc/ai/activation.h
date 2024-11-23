#pragma once

#include "matrix/matrix.h"

typedef struct
{
    double (*calculate)(double val);
    double (*derivetive)(double val);
    const char* name;
    void* params;
} ai_activation;

extern ai_activation activation_tanh; 
extern ai_activation activation_relu;

/**
 * Find activation from name. Defaults to `tanh`.
*/
ai_activation ai_activation_find(const char* name);

/**
 * Applies activation for each column of a matrix
*/
void ai_activation_apply_matrix_column(matrix mat, ai_activation activation);
