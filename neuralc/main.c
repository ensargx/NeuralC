#include "util/logger.h"
#include "matrix/matrix.h"

int main()
{
    log_debug("Testing...");

    matrix mat;

    matrix_init(&mat, 5, 10);

    matrix_set(&mat, 1, 10, 3.14);
    
    log_debug("value: %lf", matrix_get(&mat, 1, 10));

    return 0;
}
