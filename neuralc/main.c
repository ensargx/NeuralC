#include "util/logger.h"
#include "matrix/matrix.h"

int main()
{
    log_debug("Testing...");

    log_message("[AAA]", "test: %s\n", "Ensar");

    matrix mat;

    matrix_init(&mat, 5, 10);

    return 0;
}
