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
    log_debug("y.shape = (%d, %d)", y_.rows, y_.cols);
    matrix_init(&y, 10, y_.cols);

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

    matrix z1 = { 0 };
    matrix z2 = { 0 };
    matrix da1 = { 0 };
    matrix da2 = { 0 };

    matrix db2 = { 0 };
    matrix db1 = { 0 };
    matrix z1T = { 0 };

    matrix w2T = { 0 };
    matrix dw2 = { 0 };
    matrix dw1 = { 0 };
    matrix_init(&dw2, 10, 32);

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

        matrix_subtract(&da2, z2, y); 
        matrix_scale(da2, 2.0);

        matrix_sigmoid_deriv(z2);
        matrix_mul(da2, z2);

        matrix_transpose(&z1T, z1);
        matrix_dot(&dw2, da2, z1T);       // dW2 = da2 * z1^T
        
        matrix_sum_rows(&db2, da2);        // db2 = sum(da2)

        // Parametre güncellemesi
        matrix_scale(dw2, -lr);
        matrix_add(w2, dw2);

        matrix_scale(db2, -lr);
        matrix_add(b2, db2);

        matrix_transpose(&w2T, w2);      // w2^T
        matrix_dot(&da1, w2T, da2);      // da1 = w2^T * da2 (gradient of a1)

        // Apply the derivative of tanh activation function
        matrix_tanh_deriv(z1);           // Apply tanh derivative to z1
        matrix_mul(da1, z1);             // da1 = da1 * tanh'(z1)

        // Compute the gradient for dw1 (derivative of W1)
        matrix_dot(&dw1, da1, xT);       // dw1 = da1 * x^T

        // Compute the gradient for db1 (sum of rows of da1)
        matrix_sum_rows(&db1, da1);      // db1 = sum(da1, axis=0)

        // Update parameters for W1 and b1
        matrix_scale(dw1, -lr);          // Scale by learning rate
        matrix_add(w1, dw1);             // Update W1 = W1 - lr * dw1

        matrix_scale(db1, -lr);          // Scale db1 by learning rate
        matrix_add(b1, db1);             // Update b1 = b1 - lr * db1


    }
}

