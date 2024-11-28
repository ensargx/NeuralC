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
    double lr = 0.01;

    matrix w1 = matrix_create_random(32, 28*28, -1, 1, 54);
    matrix b1 = matrix_create_random(32, 1, -1, 1, 55);
    
    matrix w2 = matrix_create_random(10, 32, -1, 1, 56);
    matrix b2 = matrix_create_random(10, 1, -1, 1, 57);

    matrix x = matrix_read_csv("data/mnist_test_x.csv", 1);
    for (int i = 0; i < x.rows; ++i)
        for (int j = 0; j < x.cols; ++j)
            matrix_set(x, i, j, matrix_get(x, i, j) / 255);

    matrix y_ = matrix_read_csv("data/mnist_test_y.csv", 1);
    // y_ -> (1, N)
    // 0..9 -> [0...1]
    matrix y;
    matrix_init(&y, 10, y_.cols);
    log_debug("y.shape = (%d, %d)", y.rows, y.cols);

    for (int i = 0; i < y.rows; ++i)
    {
        for (int j = 0; j < y.cols; ++j)
        {
            matrix_set(y, i, j, 0);
            double val = matrix_get(y_, 0, j);
            int idx = (int)(val);
            matrix_set(y, idx, j, 1);
        }
    }

    matrix xT = { 0 };
    matrix_transpose(&xT, x);

    matrix y_yhat = { 0 };
    matrix_init(&y_yhat, y.rows, y.cols);

    matrix dLdy;
    matrix_init(&dLdy, y.rows, 1);

    matrix z1 = { 0 };
    matrix z2 = { 0 };
    matrix da1 = { 0 };
    matrix da2 = { 0 };

    matrix db2 = { 0 };
    matrix_init(&db2, b2.rows, b2.cols);
    matrix db1 = { 0 };
    matrix z1T = { 0 };

    matrix w2T = { 0 };
    matrix dw2 = { 0 };
    matrix_init(&dw2, w2.rows, w2.cols);
    matrix dw1 = { 0 };

    matrix sigmoid_deriv = { 0 };

    for (int iter = 0; iter < itercnt; ++iter)
    {
        matrix_dot(&z1, w1, x);
        matrix_add_row(z1, b1);
        matrix_tanh(z1);

        log_debug("shape a1 = (%d, %d)", z1.rows, z1.cols);

        matrix_dot(&z2, w2, z1);
        matrix_add_row(z2, b2);
        matrix_sigmoid(z2);

        log_debug("shape a2 = (%d, %d)", z2.rows, z2.cols);

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

        // y - y^ 
        for (int i = 0; i < y.rows; ++i)
        {
            double sum = 0;
            for (int j = 0; j < y.cols; ++j)
            {
                sum += matrix_get(z2, i, j) - matrix_get(y, i, j);
            }
            matrix_set(dLdy, i, 0, sum / y.cols);
        }

        log_debug("shape dLdy: (%d, %d)", dLdy.rows, dLdy.cols);

        matrix_transpose(&z1T, z1);

        // (y-y^)*(sigmoid'(z))
        matrix_sigmoid_deriv(&sigmoid_deriv, z2);
        matrix_mul(y_yhat, sigmoid_deriv);
        matrix_dot(&dw2, y_yhat, z1T);
        // y_yhat = da2,
        // dw2 = dw2

        // dLdy * dy/dw2 = dL/dw2
        for (int i = 0; i < dLdy.rows; ++i)
        {
            for (int j = 0; j < dw2.cols; ++j)
            {
                double val = matrix_get(dw2, i, j) * matrix_get(dLdy, i, 0);
                matrix_set(dw2, i, j, val);
            }
        }
        // dL/db2 = dLdy * dy/db2
        for (int i = 0; i < db2.rows; ++i)
        {
            for (int j = 0; j < db2.cols; ++j)
            {
                double val = matrix_get(y_yhat, i, j) * matrix_get(dLdy, i, 0);
                matrix_set(db2, i, j, val);
            }
        }

        log_debug("dw2.shape = (%d, %d)", dw2.rows, dw2.cols);
        log_debug("db2.shape = (%d, %d)", db2.rows, db2.cols);

        // dL/dw1 = 

        // update w2
        // fuck w1, later i update it.
        // w2 -= lr * dw2
        for (int i = 0; i < w2.rows; ++i)
        {
            for(int j = 0; j < w2.cols; ++j)
            {
                double new = -lr * matrix_get(dw2, i, j);
                matrix_set(w2, i, j, new);
            }
        }

        for (int i = 0; i < b2.rows; ++i)
        {
            for (int j = 0; j < b2.cols; ++j)
            {
                double new = -lr * matrix_get(db2, i, j);
                matrix_set(b2, i, j, new);
            }
        }

    }
}

