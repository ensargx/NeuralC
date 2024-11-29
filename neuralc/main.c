
#include "matrix/matrix.h"
#include "util/logger.h"
#include <math.h>

void gd(void);
void sgd(void);
void adam(void);

int main(void)
{
    adam();
}

void sgd(void)
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

void gd(void)
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
    matrix db = { 0 };
    matrix_init(&db, b.rows, b.cols);

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

        matrix_zero(dw);
        matrix_zero(db);

        matrix_scale(&da, dz, d_cost);

        for (int k = 0; k < a.cols; ++k)
        {
            for (int i = 0; i < w.rows; ++i)
            {
                for (int j = 0; j < w.cols; ++j)
                {
                    double sum = x.data[j][k] * da.data[i][k];
                    dw.data[i][j] += sum;
                }
                db.data[i][0] += da.data[i][k];
            }
        }

        for(int i = 0; i < w.rows; ++i)
        {
            for (int j = 0; j < w.cols; ++j)
            {
                dw.data[i][j] /= a.cols;
                w.data[i][j] -= lr * dw.data[i][j];
            }
            db.data[i][0] /= a.cols;
            b.data[i][0] -= lr * db.data[i][0];
        }

        log_debug("COST: %lf", cost);

    }
}

void adam(void)
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

    matrix z = { 0 };
    matrix a = { 0 };

    matrix dz = { 0 };
    matrix da = { 0 };

    matrix m = { 0 };
    matrix_init(&m, w.rows, w.cols + 1);
    matrix_zero(m);

    matrix mt = { 0 };
    matrix_init(&mt, w.rows, w.cols + 1);
    matrix_zero(mt);

    matrix v = { 0 };
    matrix_init(&v, w.rows, w.cols + 1);
    matrix_zero(v);

    matrix vt = { 0 };
    matrix_init(&vt, w.rows, w.cols + 1);
    matrix_zero(vt);

    int t = 0;

    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 0.00000001;
    double alpha = 0.001;

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

        t += 1;

        for(int i = 0; i < w.rows; ++i)
        {
            for(int j = 0; j < w.cols; ++j)
            {
                double sum = x.data[j][dataidx] * da.data[0][dataidx];

                m.data[i][j] = beta1 * m.data[i][j] + (1-beta1) * sum;
                v.data[i][j] = beta2 * v.data[i][j] + (1-beta2) * sum * sum;

                mt.data[i][j] = m.data[i][j] / (1-pow(beta1, t));
                vt.data[i][j] = v.data[i][j] / (1-pow(beta2, t));
            }

            m.data[i][w.cols] = beta1 * m.data[i][w.cols] + (1-beta1) * da.data[0][dataidx];
            v.data[i][w.cols] = beta2 * v.data[i][w.cols] + (1-beta2) * da.data[0][dataidx] * da.data[0][dataidx];

            mt.data[i][w.cols] = m.data[i][w.cols] / (1-pow(beta1, t));
            vt.data[i][w.cols] = v.data[i][w.cols] / (1-pow(beta2, t));
        }

        for (int i = 0; i < w.rows; ++i)
        {
            for (int j = 0; j < w.cols; ++j)
            {
                w.data[i][j] -= alpha * mt.data[i][j] / (pow(vt.data[i][j], 0.5) + epsilon);
            }
            b.data[i][0] -= alpha * mt.data[i][w.cols] / (pow(vt.data[i][w.cols], 0.5) + epsilon);
        }

        log_debug("cost: %lf", cost);
    }
}
