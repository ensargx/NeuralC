#include "util/logger.h"
#include "matrix/matrix.h"

int main()
{
    log_debug("Testing...");

    matrix mat;

    matrix_init(&mat, 5, 10);

    return 0;
}
