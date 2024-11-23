#pragma once

#include "layers.h"
#include "activation.h"

/**
 * Every AI model should be respesentable in binary.
 * Also serializable in text and can be re-produced from a file.
 *
 * Each model will have `activation`, `layers`.
*/

typedef struct
{
    ai_activation activation;
    ai_layers layers;
} ai_model;
