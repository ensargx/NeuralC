
#include "matrix/matrix.h"
#include "util/logger.h"

int main(void)
{
    int n = 28;
    matrix x_ = matrix_read_csv("data/data_x.csv", 1);
    matrix x = { 0 }; // (728, N)
    matrix_transpose(&x, x_);
    for (int i = 0; i < x.rows; ++i)
    {
        for (int j = 0; j < x.cols; ++j)
        {
            x.data[i][j] = x.data[i][j] / 255;
        }
    }
    log_debug("x.shape = (%d, %d)", x.rows, x.cols);

    matrix y_ = matrix_read_csv("data/data_y.csv", 1);
    matrix y = { 0 };
    matrix_transpose(&y, y_);

    matrix w = matrix_create_random(1, n*n, -1, 1, 54);
    matrix b = matrix_create_random(1, 1, -1, 1, 1);

    int itercnt = 1000;
    double lr = 0.01;

    matrix z = { 0 };
    matrix a = { 0 };

    matrix dz = { 0 };
    matrix da = { 0 };

    matrix dw = { 0 };
    matrix_init(&dw, w.rows, w.cols);

    for (int iter = 0; iter < itercnt; ++iter)
    {
        matrix_dot(&z, w, x);
        matrix_add_row(z, b);
        matrix_tanh(&a, z);

        double d_cost = 0;
        double cost = 0;
        for (int i = 0; i < a.cols; ++i)
        {
            double y_hat = a.data[0][i];
            double expected = y_.data[i][0];
            d_cost += (y_hat - expected);
            cost += (expected - y_hat) * (expected - y_hat);
        }
        d_cost /= a.cols;
        cost /= a.cols;

        matrix_tanh_deriv(&dz, z);

        matrix_scale(&da, dz, d_cost);

        int dataidx = iter % x.rows;

        for (int i = 0; i < w.rows; ++i)
        {
            for (int j = 0; j < w.cols; ++j)
            {
                double sum = x.data[j][dataidx] * da.data[0][dataidx];
                w.data[i][j] -= lr * sum;
            }
            b.data[i][0] -= lr * da.data[0][dataidx];
        }

        log_debug("COST: %lf", cost);

    }
}
