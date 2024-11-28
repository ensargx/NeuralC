#include "matrix/matrix.h"
#include "util/logger.h"
#include <math.h>

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

// 
// neural with 
// w1 = (32, 28*28)
// b1 = (1, 32)
// z1 = w1 * x + b1
// a1 = tanh(z1)
//
// w2 = (10, 32)
// b2 = (1, 10)
// z2 = w2 * a1 + b2
// a2 = tanh(z1)
//

void test3(void)
{



    int itercnt = 10000;
    double lr = 0.0001;
    int seed = 312;

    matrix w1 = matrix_create_random(32, 28*28, -1, 1, seed);
    matrix b1 = matrix_create_random(32, 1, -1, 1, seed+1);
    
    matrix w2 = matrix_create_random(10, 32, -1, 1, seed+2);
    matrix b2 = matrix_create_random(10, 1, -1, 1, seed+3);

    matrix x_ = matrix_read_csv("data/data_x.csv", 1);
    for (int i = 0; i < x_.rows; ++i)
        for (int j = 0; j < x_.cols; ++j)
            matrix_set(x_, i, j, matrix_get(x_, i, j) / 255);

    matrix y_ = matrix_read_csv("data/data_y.csv", 1);

    matrix x = { 0 };
    matrix y = { 0 };
    matrix_transpose(&x, x_);
    matrix_transpose(&y, y_);

    log_debug("x.shape = (%d, %d)", x.rows, x.cols);
    log_debug("y.shape = (%d, %d)", y.rows, y.cols);

    matrix xT = { 0 };
    matrix_transpose(&xT, x);


    matrix z1 = { 0 };
    matrix z2 = { 0 };

    matrix dL_da2 = { 0 };
    matrix_init(&dL_da2, y.rows, 1);

    matrix a1 = { 0 };
    matrix a2 = { 0 };

    matrix a1T = { 0 };
    matrix w2T = { 0 };

    matrix dL_dw1 = { 0 };
    matrix dL_dw2 = { 0 };
    matrix dL_db1;
    matrix_init(&dL_db1, b1.rows, b1.cols);
    matrix dL_db2;
    matrix_init(&dL_db2, b2.rows, b2.cols);

    matrix da2_dw1 = { 0 };
    matrix da2_dw2 = { 0 };
    matrix sigmoid_deriv = { 0 };
    matrix tanh_deriv = { 0 };

    for (int iter = 0; iter < itercnt; ++iter)
    {
        matrix_dot(&z1, w1, x);
        matrix_add_row(z1, b1);
        matrix_copy(&a1, z1);
        matrix_tanh(a1);

        matrix_dot(&z2, w2, a1);
        matrix_add_row(z2, b2);
        matrix_copy(&a2, z2);
        matrix_sigmoid(a2);

        log_debug("prediction for %d", iter);
        for (int i = 0; i < z2.rows; ++i)
        {
            log_debug("a2[%d][%d] = %lf", i, iter, matrix_get(z2, i, iter));
            log_debug("y[%d][%d] = %lf", i, iter, matrix_get(y, i, iter));
        }

        // calculate loss
        double loss = 0;
        for (int i = 0; i < z2.rows; ++i)
        {
            // i for all 10 result
            double loss_ = 0;
            for (int j = 0; j < z2.cols; ++j)
            {
                loss_ += pow(matrix_get(z2, i, j) - matrix_get(y, i, j), 2);
            }
            loss_ /= z2.cols;
            loss += loss_;
        }
        log_debug("loss: %lf", loss);

        double C = 0;
        // y - y^
        for (int i = 0; i < y.rows; ++i)
        {
            double sum = 0;
            for (int j = 0; j < y.cols; ++j)
            {
                sum += -2 * (matrix_get(y, i, j) - matrix_get(z2, i, j));
                C += -2 * (matrix_get(y, i, j) - matrix_get(z2, i, j));
            }
            matrix_set(dL_da2, i, 0, sum);
        }
        C /= y.rows * y.cols;

        matrix_transpose(&a1T, a1);
        matrix_transpose(&w2T, w2);

        // (y-y^)*(sigmoid'(z))
        matrix_sigmoid_deriv(&sigmoid_deriv, z2); // (10, N)

        // dLdy * dy/dw2 = dL/dw2
        // dL/dw2 = dL/dy * dy/dw2
        //         (y-y^) * (sigmoid')
        matrix_dot(&da2_dw2, sigmoid_deriv, a1T); // (10, 32)
        matrix_scale(&dL_dw2, da2_dw2, C); // (10, 32)
        matrix_scale(&dL_db2, sigmoid_deriv, C); // (10, N)

        matrix_tanh_deriv(&tanh_deriv, z1); // 32 * N
        matrix_dot(&da2_dw1, tanh_deriv, xT);
        matrix_scale(&dL_dw1, da2_dw1, C);

        matrix_scale(&dL_db1, tanh_deriv, C);

        // update the parameters.

        for (int i = 0; i < w2.rows; ++i)
        {
            for(int j = 0; j < w2.cols; ++j)
            {
                double new = -lr * matrix_get(dL_dw2, i, j);
                matrix_set(w2, i, j, new);
            }
        }

        for (int i = 0; i < b2.rows; ++i)
        {
            for (int j = 0; j < b2.cols; ++j)
            {
                double new = -lr * matrix_get(dL_db2, i, j);
                matrix_set(b2, i, j, new);
            }
        }

        for (int i = 0; i < w1.rows; ++i)
        {
            for (int j = 0; j < w1.cols; ++j)
            {
                double new = -lr * matrix_get(dL_dw1, i, j);
                matrix_set(w1, i, j, new);
            }
        }

        for (int i = 0; i < b1.rows; ++i)
        {
            for (int j = 0; j < b1.cols; ++j)
            {
                double new = -lr * matrix_get(dL_db1, i, j);
                matrix_set(b1, i, j, new);
            }
        }

    }
}

