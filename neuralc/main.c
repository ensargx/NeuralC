#include "matrix/matrix.h"
#include "util/logger.h"

double sigmoid_deriv(double);
void test2();

int main()
{
    test2();
    return 0;
    int n = 28;

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

    // 0-1 trans
    for (int i = 0; i < x.rows; ++i)
        for (int j = 0; j < x.cols; ++j)
            matrix_set(x, i, j, matrix_get(x, i, j) / 255);

    matrix y = matrix_read_csv("data/mnist_test_y.csv", 1);
    log_debug("y matrix: row count: %d, column count: %d", y.rows, y.cols);

    //      (32xn*n) * ( n*n, A ) + (32, 1) -> (32, A)
    // z1 = W1 * x + b1 = ( 32, A )
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

    // a2: 10 * 10000
    log_debug("a2 rows: %d, cols: %d", a2.rows, a2.cols);

    // calculate loss
    double cost = 0;
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
        cost += l;
    }
    cost /= a2.rows;

    log_debug("cost: %lf", cost);

    // dC/dw2 = dz2/dw2 * da/dz2 * dC/da2
    // dC/db1 = dz2/db2 * da/dz2 * dC/da2 
    
    matrix dC_dw2;
    matrix dC_db2;
    matrix_init(&dC_dw2, a2.rows, 1);
    matrix_init(&dC_db2, a2.rows, 1);
    for (int i = 0; i < a2.rows; ++i)
    {
        double l = 0;
        double lb = 0;
        for (int j = 0; j < a2.cols; ++j)
        {
            double real_yencoded = matrix_get(y, 0, j);
            double real_y = 0;
            if ( real_yencoded == i )
                real_y = 1;
            double dz_dw = matrix_get(a2, i, j);
            double da_dz = sigmoid_deriv(matrix_get(z2, i, j));
            double dc_da = matrix_get(a2, i, j) - real_y;
            l = 2 * dz_dw * da_dz * dc_da;
            lb = 2 * da_dz * dc_da;
        }
        matrix_set(dC_dw2, i, 0, l / a2.rows);
        matrix_set(dC_db2, i, 0, lb / a2.rows);
    }
    for (int i = 0; i < dC_dw2.rows; ++i)
        log_debug("dCda2[%d][%d] = %lf", i, 0, matrix_get(dC_dw2, i, 0));

}
