#pragma once

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
 * Find activation from name.
*/
ai_activation ai_activation_find(const char* name);
