#include "matrix/matrix.h"
#include "util/logger.h"

void test2(void)
{
    int n = 28;

    matrix w = matrix_create_random(1, n*n, -1, 1, 54); 
    matrix b = matrix_create_random(1, 1, -1, 1, 55);
    
    matrix x = matrix_read_csv("data/filtered_data_x.csv", 1);
    matrix y = matrix_read_csv("data/filtered_data_y.csv", 1);

    for (int i = 0; i < x.rows; ++i)
        for (int j = 0; j < x.cols; ++j)
            matrix_set(x, i, j, matrix_get(x, i, j) / 255);

    log_debug("x shape: (%d x %d)", x.rows, x.cols);
    log_debug("y shape: (%d x %d)", y.rows, y.cols);

    matrix z = { 0 };
    int iteration = 10000;
    double lr = 0.01;

    for (int iter = 0; iter < iteration; ++iter)
    {
        matrix_dot(&z, w, x);
        matrix_add_row(z, b);
        matrix_tanh(z);

        double loss = 0;
        for (int i = 0; i < z.cols; i++)
        {
            double true_y = matrix_get(y, 0, i);
            double y_hat = matrix_get(z, 0, i);
            double sq = true_y - y_hat;
            loss += sq * sq;
        }
        loss /= z.cols;
        log_debug("loss: %lf", loss);

        for (int i = 0; i < w.cols; ++i)
        {
            double grad_w = 0;
            for (int j = 0; j < z.cols; ++j)
            {
                double true_y = matrix_get(y, 0, j);
                double y_hat = matrix_get(z, 0, j);
                double dz = 2 * (y_hat - true_y);
                double tanh_derivative = 1 - matrix_get(z, 0, j) * matrix_get(z, 0, j);
                grad_w += dz * tanh_derivative * matrix_get(x, i, j);
            }
            grad_w /= z.cols;

            double old_w = matrix_get(w, 0, i);
            double new_w = old_w - lr * grad_w;
            matrix_set(w, 0, i, new_w);
        }

    }

}
