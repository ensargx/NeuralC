#include "matrix/matrix.h"
#include "util/logger.h"

int main()
{
    int n = 28;
    matrix w1;
    matrix b1;
    matrix_init(&w1, 32, n*n+1);
    matrix_init(&b1, 1, n*n+2);

    matrix w2;
    matrix b2;
    matrix_init(&w2, 10, 32);
    matrix_init(&b2, 1, 32);

    matrix x = matrix_read_csv("data/mnist_test.csv", 1);
    log_debug("x matrix: row count: %d, column count: %d", x.rows, x.cols);

    // z1 = W1 * x + b1
    matrix z1 = { 0 };
    matrix_dot(&z1, w1, x);
    matrix_add_row(z1, b1);
    // a1 = tanh(z1)
    matrix a1 = z1;
    matrix_tanh(a1);

    // z2 = W2 * a1 + b2
    matrix z2 = { 0 };
    matrix_dot(&z2, w2, a1);
    matrix_add_row(z2, b2);
    // a2 = sigmoid(z2)
    matrix a2 = z2;
    matrix_sigmoid(a2);

}
