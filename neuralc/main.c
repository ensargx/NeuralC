#include "matrix/matrix.h"
#include "util/logger.h"
#include <signal.h>
#include <stdlib.h>

int main()
{
    int n = 28;
    /*
    matrix w1;
    matrix b1;
    matrix_init(&w1, 32, n*n+1);
    matrix_init(&b1, 1, n*n+2);
    */

    matrix w1 = matrix_create_random(32, n*n, -1, 1, 54);
    matrix b1 = matrix_create_random(32, 1, -1, 1, 55);

    matrix w2 = matrix_create_random(10, 32, -1, 1, 56);
    matrix b2 = matrix_create_random(10, 1, -1, 1, 57);

    /*
    matrix_init(&w2, 10, 32);
    matrix_init(&b2, 1, 32);
    */

    matrix x = matrix_read_csv("data/mnist_test_x.csv", 1);
    log_debug("x matrix: row count: %d, column count: %d", x.rows, x.cols);

    matrix y = matrix_read_csv("data/mnist_test_y.csv", 1);
    log_debug("y matrix: row count: %d, column count: %d", y.rows, y.cols);

    //      (32xn*n) * ( n*n, A ) + ( 1, 32 )
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

    log_debug("a2 rows: %d, cols: %d", a2.rows, a2.cols);

    // calculate loss
    matrix loss;
    matrix_init(&loss, 1, a2.rows);
    for (int i = 0; i < a2.rows; ++i)
    {
        double l = 0;
        for (int j = 0; j < a2.cols; ++j)
        {
            double y_hat = matrix_get(a2, i, j);
            double real_yencoded = matrix_get(y, 0, j);
            double real_y = 0;
            if ( real_yencoded == i )
                real_y = 1;
            l += (y_hat - real_y) * (y_hat - real_y);
        }
        matrix_set(loss, 0, i, l / x.rows);
    }

    // print loss
    for (int i = 0; i < loss.rows; ++i)
        for(int j = 0; j < loss.cols; ++j)
            log_debug("loss[%d][%d] = %lf", i, j, matrix_get(loss, i, j));

}
