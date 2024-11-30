#include "matrix/matrix.h"
#include "util/logger.h"

#include <math.h>
#include <stdio.h>
#include <time.h>

void sgd(matrix x, matrix y, matrix w, matrix b, const char* param_name, matrix x_test, matrix y_test);
void gd(matrix x, matrix y, matrix w, matrix b, const char* param_name, matrix x_test, matrix y_test);
void adam(matrix x, matrix y, matrix w, matrix b, const char* param_name, matrix x_test, matrix y_test);
void params_to_csv(FILE* file, matrix w, matrix b, int iter, double cost, double elapsed);
double check_correct(matrix w, matrix b, matrix x_test, matrix y_test);

const int BATCH_SIZE = 1000;
const int EPOCHS = 1000;

int main(void)
{
    const char* xpath = "data/train_data_x.csv";
    const char* ypath = "data/train_data_y.csv";
    
    const char* xtestpath = "data/test_data_x.csv";
    const char* ytestpath = "data/test_data_y.csv";

    matrix x_ = matrix_read_csv(xpath, 1);
    matrix x = { 0 }; // (728, N)
    matrix_transpose(&x, x_);
    matrix_destroy(&x_);
    log_debug("x.shape = (%d, %d)", x.rows, x.cols);

    matrix y_ = matrix_read_csv(ypath, 1);
    matrix y = { 0 };
    matrix_transpose(&y, y_);
    matrix_destroy(&y_);
    log_debug("y.shape = (%d, %d)", y.rows, y.cols);

    matrix x_test_ = matrix_read_csv(xtestpath, 1);
    matrix y_test_ = matrix_read_csv(ytestpath, 1);
    matrix x_test = { 0 };
    matrix y_test = { 0 };
    matrix_transpose(&x_test, x_test_);
    matrix_transpose(&y_test, y_test_);
    matrix_destroy(&x_test_);
    matrix_destroy(&y_test_);

    int n = 28;

    matrix w = matrix_create_random(1, n*n, -1, 1, 54);
    matrix b = matrix_create_random(1, 1, -1, 1, 1);
    adam(x, y, w, b, "54", x_test, y_test);
    return 0;

    gd(x, y, w, b, "54", x_test, y_test);
    sgd(x, y, w, b, "54", x_test, y_test);
    adam(x, y, w, b, "54", x_test, y_test);

    matrix_destroy(&w);
    matrix_destroy(&b);
    w = matrix_create_random(1, n*n, -1, 1, 65);
    b = matrix_create_random(1, 1, -1, 1, 1);
    gd(x, y, w, b, "65", x_test, y_test);
    sgd(x, y, w, b, "65", x_test, y_test);
    adam(x, y, w, b, "65", x_test, y_test);

    matrix_destroy(&w);
    matrix_destroy(&b);
    w = matrix_create_random(1, n*n, -1, 1, 98);
    b = matrix_create_random(1, 1, -1, 1, 1);
    gd(x, y, w, b, "98", x_test, y_test);
    sgd(x, y, w, b, "98", x_test, y_test);
    adam(x, y, w, b, "98", x_test, y_test);

    matrix_destroy(&w);
    matrix_destroy(&b);
    w = matrix_create_random(1, n*n, -1, 1, 120);
    b = matrix_create_random(1, 1, -1, 1, 1);
    gd(x, y, w, b, "120", x_test, y_test);
    sgd(x, y, w, b, "120", x_test, y_test);
    adam(x, y, w, b, "120", x_test, y_test);

    matrix_destroy(&w);
    matrix_destroy(&b);
    w = matrix_create_random(1, n*n, -1, 1, 10);
    b = matrix_create_random(1, 1, -1, 1, 1);
    gd(x, y, w, b, "10", x_test, y_test);
    sgd(x, y, w, b, "10", x_test, y_test);
    adam(x, y, w, b, "10", x_test, y_test);

    matrix_destroy(&w);
    matrix_destroy(&b);
    matrix_destroy(&x);
    matrix_destroy(&y);
}

void sgd(matrix x, matrix y, matrix w_, matrix b_, const char* param_name, matrix x_test, matrix y_test)
{
    double lr = 0.01;

    clock_t start, end;
    double elapsed;
    start = clock();

    matrix w = { 0 };
    matrix_copy(&w, w_);
    matrix b = { 0 };
    matrix_copy(&b, b_);

    matrix z = { 0 };
    matrix a = { 0 };

    matrix dz = { 0 };
    matrix da = { 0 };

    char outdir[64] = { 0 };
    sprintf(outdir, "out/sgd_%s.csv", param_name);
    FILE *out_params = fopen(outdir, "w");

    for (int iter = 0; iter < EPOCHS; ++iter)
    {
        matrix_dot(&z, w, x);
        matrix_add_row(z, b);
        matrix_tanh(&a, z);

        double d_cost = 0;
        double cost = 0;
        for (int i = 0; i < a.cols; ++i)
        {
            double y_hat = a.data[0][i];
            double expected = y.data[0][i];
            d_cost += (y_hat - expected);
            cost += (expected - y_hat) * (expected - y_hat);
        }
        d_cost /= a.cols;
        cost /= a.cols;

        matrix_tanh_deriv(&dz, z);

        matrix_scale(&da, dz, d_cost);

        double correct = check_correct(w, b, x_test, y_test);

        end = clock();
        elapsed = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;

        params_to_csv(out_params, w, b, iter, cost, elapsed);

        int dataidx = iter % x.rows;

        for (int i = 0; i < w.rows; ++i)
        {
            for (int j = 0; j < w.cols; ++j)
            {
                double sum = x.data[j][dataidx] * da.data[i][dataidx];
                w.data[i][j] -= lr * sum;
            }
            b.data[i][0] -= lr * da.data[i][dataidx];
        }

        log_debug("ITER: %d, COST SGD: %lf, CORRECT: %lf", iter, cost, correct);

    }

    fclose(out_params);

    matrix_destroy(&w);
    matrix_destroy(&b);
    matrix_destroy(&z);
    matrix_destroy(&a);
    matrix_destroy(&dz);
    matrix_destroy(&da);
}

void gd(matrix x, matrix y, matrix w_, matrix b_, const char* param_name, matrix x_test, matrix y_test)
{
    double lr = 0.01;

    clock_t start, end;
    double elapsed;
    start = clock();

    matrix w = { 0 };
    matrix_copy(&w, w_);
    matrix b = { 0 };
    matrix_copy(&b, b_);

    matrix z = { 0 };
    matrix a = { 0 };

    matrix dz = { 0 };
    matrix da = { 0 };

    matrix dw = { 0 };
    matrix_init(&dw, w.rows, w.cols);
    matrix db = { 0 };
    matrix_init(&db, b.rows, b.cols);

    char outdir[64] = { 0 };
    sprintf(outdir, "out/gd_%s.csv", param_name);
    FILE *out_params = fopen(outdir, "w");

    for (int iter = 0; iter < EPOCHS; ++iter)
    {
        matrix_dot(&z, w, x);
        matrix_add_row(z, b);
        matrix_tanh(&a, z);

        double d_cost = 0;
        double cost = 0;
        for (int i = 0; i < a.cols; ++i)
        {
            double y_hat = a.data[0][i];
            double expected = y.data[0][i];
            d_cost += (expected - y_hat);
            cost += (expected - y_hat) * (expected - y_hat);
        }
        d_cost /= a.cols;
        cost /= a.cols;

        matrix_tanh_deriv(&dz, z);

        matrix_zero(dw);
        matrix_zero(db);

        double correct = check_correct(w, b, x_test, y_test);

        matrix_scale(&da, dz, d_cost);

        end = clock();
        elapsed = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;

        params_to_csv(out_params, w, b, iter, cost, elapsed);

        for (int k_ = iter * BATCH_SIZE; k_ < (iter+1)*BATCH_SIZE; ++k_)
        {
            int k = k_ % a.cols;
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
                dw.data[i][j] /= BATCH_SIZE;
                w.data[i][j] -= lr * dw.data[i][j];
            }
            db.data[i][0] /= BATCH_SIZE;
            b.data[i][0] -= lr * db.data[i][0];
        }

        log_debug("ITER: %d, COST GD: %lf, CORRECT: %lf", iter, cost, correct);

    }

    fclose(out_params);

    matrix_destroy(&w);
    matrix_destroy(&b);
    matrix_destroy(&z);
    matrix_destroy(&a);
    matrix_destroy(&dz);
    matrix_destroy(&da);
    matrix_destroy(&dw);
    matrix_destroy(&db);
}

void adam(matrix x, matrix y, matrix w_, matrix b_, const char* param_name, matrix x_test, matrix y_test)
{
    clock_t start, end;
    double elapsed;
    start = clock();

    matrix w = { 0 };
    matrix_copy(&w, w_);
    matrix b = { 0 };
    matrix_copy(&b, b_);

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

    char outdir[64] = { 0 };
    sprintf(outdir, "out/adam_%s.csv", param_name);
    FILE *out_params = fopen(outdir, "w");

    for (int iter = 0; iter < EPOCHS; ++iter)
    {
        matrix_dot(&z, w, x);
        matrix_add_row(z, b);
        matrix_tanh(&a, z);

        double d_cost = 0;
        double cost = 0;
        for (int i = 0; i < a.cols; ++i)
        {
            double y_hat = a.data[0][i];
            double expected = y.data[0][i];
            d_cost += (expected - y_hat);
            cost += pow((expected - y_hat), 2);
        }
        d_cost /= a.cols;
        cost /= a.cols;

        matrix_tanh_deriv(&dz, z);

        matrix_scale(&da, dz, d_cost);

        double correct = check_correct(w, b, x_test, y_test);

        end = clock();
        elapsed = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;

        params_to_csv(out_params, w, b, iter, cost, elapsed);

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

        log_debug("ITER: %d, COST ADAM: %lf, CORRECT: %lf", iter, cost, correct);
    }

    fclose(out_params);

    matrix_destroy(&w);
    matrix_destroy(&b);
    matrix_destroy(&z);
    matrix_destroy(&a);
    matrix_destroy(&dz);
    matrix_destroy(&da);
    matrix_destroy(&m);
    matrix_destroy(&mt);
    matrix_destroy(&v);
    matrix_destroy(&vt);
}

void params_to_csv(FILE *file, matrix w, matrix b, int iter, double cost, double elapsed)
{
    fprintf(file, "%d,%lf,%lf,", iter, cost, elapsed);
    for (int i = 0; i < w.cols; ++i)
        fprintf(file, "%.3lf,", w.data[0][i]);
    fprintf(file, "%.3lf\n", b.data[0][0]);
}

double check_correct(matrix w, matrix b, matrix x_test, matrix y_test)
{
    matrix z = { 0 };
    matrix a = { 0 };

    matrix_dot(&z, w, x_test);
    matrix_add_row(z, b);
    matrix_tanh(&a, z);

    int correct = 0;
    for (int i = 0; i < y_test.cols; ++i)
    {
        double predicted = a.data[0][i];
        if ( predicted > 1 || predicted < -1 )
            log_error("error predict: %lf", predicted);

        if ( predicted > 0 )
        {
            if ( y_test.data[0][i] > 0 )
                correct++;
        }
        else 
        {
            if ( y_test.data[0][i] < 0 )
                correct++;
        }
    }

    matrix_destroy(&z);
    matrix_destroy(&a);

    return (double)correct / y_test.cols;
}
