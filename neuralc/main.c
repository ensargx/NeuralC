#include "matrix/matrix.h"
#include "util/logger.h"

#include <math.h>
#include <stdio.h>
#include <time.h>

void sgd(matrix x, matrix y, matrix w, const char* param_name, matrix x_test, matrix y_test);
void gd(matrix x, matrix y, matrix w, const char* param_name, matrix x_test, matrix y_test);
void adam(matrix x, matrix y, matrix w, const char* param_name, matrix x_test, matrix y_test);
void params_to_csv(FILE* file, matrix w, int iter, double cost, double elapsed, double correct);
double check_correct(matrix w, matrix x_test, matrix y_test);

const int BATCH_SIZE = 1000;
const int EPOCHS = 500;

int main(void)
{
    const char* xpath = "data/data_train_x.csv";
    const char* ypath = "data/data_train_y.csv";
    
    const char* xtestpath = "data/data_test_x.csv";
    const char* ytestpath = "data/data_test_y.csv";

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
    matrix w = { 0 };

    int rowcnt = y.rows;
    int colcnt = n*n+1; // == x.rows

    matrix_create_random(&w, rowcnt, colcnt, -1, 1, 54);
    gd(x, y, w, "54", x_test, y_test);
    sgd(x, y, w, "54", x_test, y_test);
    adam(x, y, w, "54", x_test, y_test);

    matrix_create_random(&w, rowcnt, colcnt, -1, 1, 65);
    gd(x, y, w, "65", x_test, y_test);
    sgd(x, y, w, "65", x_test, y_test);
    adam(x, y, w, "65", x_test, y_test);

    matrix_create_random(&w, rowcnt, colcnt, -1, 1, 98);
    gd(x, y, w, "98", x_test, y_test);
    sgd(x, y, w, "98", x_test, y_test);
    adam(x, y, w, "98", x_test, y_test);

    matrix_create_random(&w, rowcnt, colcnt, -1, 1, 120);
    gd(x, y, w, "120", x_test, y_test);
    sgd(x, y, w, "120", x_test, y_test);
    adam(x, y, w, "120", x_test, y_test);

    matrix_create_random(&w, rowcnt, colcnt, -1, 1, 10);
    gd(x, y, w, "10", x_test, y_test);
    sgd(x, y, w, "10", x_test, y_test);
    adam(x, y, w, "10", x_test, y_test);

    matrix_destroy(&w);
    matrix_destroy(&x);
    matrix_destroy(&y);
}

void sgd(matrix x, matrix y, matrix w_, const char* param_name, matrix x_test, matrix y_test)
{
    double lr = 0.01;

    clock_t start, end;
    double elapsed = 0;
    double totaltime = 0;

    matrix w = { 0 };
    matrix_copy(&w, w_);

    matrix z = { 0 };
    matrix a = { 0 };

    matrix dz = { 0 };
    matrix da = { 0 };
    matrix_init(&da, w.rows, w.cols);

    matrix d_cost = { 0 };
    matrix_init(&d_cost, 1, w.rows);

    char outdir[64] = { 0 };
    sprintf(outdir, "out/sgd_%s.csv", param_name);
    FILE *out_params = fopen(outdir, "w");

    for (int iter = 0; iter < EPOCHS; ++iter)
    {
        matrix_dot(&z, w, x);
        matrix_tanh(&a, z);

        int dataidx = iter % x.rows;

        double cost = 0;
        for (int i = 0; i < a.cols; ++i)
        {
            double y_hat = a.data[0][i];
            double expected = y.data[0][i];
            double diff = y_hat - expected;
            cost += pow(diff, 2);
        }
        cost /= a.cols;

        start = clock();

        double correct = check_correct(w, x_test, y_test);

        params_to_csv(out_params, w, iter, cost, totaltime, correct);

        for (int i = 0; i < d_cost.cols; ++i)
        {
            d_cost.data[0][i] = 2 * ( a.data[i][dataidx] - y.data[i][dataidx] );
        }

        matrix_tanh_deriv(&dz, z);

        for (int i = 0; i < d_cost.cols; ++i)
        {
            double d_costv = 2 * ( matrix_get(a, i, dataidx) /* a.data[i][dataidx] */ - matrix_get(y, i, dataidx) /* y.data[i][dataidx] */ );
            matrix_set(da, i, dataidx, d_costv * matrix_get(dz, i, dataidx));
            // da.data[i][dataidx] = dz.data[i][dataidx] * d_costv;
        }
        // matrix_scale(&da, dz, d_cost);

        for (int i = 0; i < w.rows; ++i)
        {
            for (int j = 0; j < w.cols; ++j)
            {
                double sum = x.data[j][dataidx] * da.data[i][dataidx];
                w.data[i][j] -= lr * sum;
            }
        }

        end = clock();
        elapsed = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
        totaltime += elapsed;

        if ( iter % 10 == 0 )
            log_debug("SGD = ITER: %d, COST: %lf, CORRECT: %lf", iter, cost, correct);

    }

    fclose(out_params);

    matrix_destroy(&w);
    matrix_destroy(&z);
    matrix_destroy(&a);
    matrix_destroy(&dz);
    matrix_destroy(&da);
    matrix_destroy(&d_cost);
}

void gd(matrix x, matrix y, matrix w_, const char* param_name, matrix x_test, matrix y_test)
{
    double lr = 0.01;

    int output_size = w_.rows;
    int num_sample = x.cols;

    clock_t start, end;
    double elapsed = 0;
    double totaltime = 0;

    matrix w = { 0 };
    matrix_copy(&w, w_);

    matrix z = { 0 };
    matrix a = { 0 };

    matrix dz = { 0 };
    matrix da = { 0 };
    matrix_init(&da, output_size, num_sample);

    matrix dw = { 0 };
    matrix_init(&dw, w.rows, w.cols);

    matrix d_cost = { 0 };
    matrix_init(&d_cost, 1, output_size);

    char outdir[64] = { 0 };
    sprintf(outdir, "out/gd_%s.csv", param_name);
    FILE *out_params = fopen(outdir, "w");

    for (int iter = 0; iter < EPOCHS; ++iter)
    {
        matrix_dot(&z, w, x);
        matrix_tanh(&a, z);

        start = clock();

        double cost = 0;
        for (int i = 0; i < a.rows; ++i)
        {
            for (int j = 0; j < a.cols; ++j)
            {
                double y_hat = a.data[0][j];
                double expected = y.data[0][j];
                double diff = y_hat - expected;
                d_cost.data[0][i] += 2*diff;
                cost += pow(diff, 2);
            }
        }
        cost /= a.cols * a.rows;

        matrix_tanh_deriv(&dz, z);
        matrix_zero(dw);

        double correct = check_correct(w, x_test, y_test);

        // dz = ( output_size, num_sample )
        for (int i = 0; i < output_size; ++i)
        {
            for (int j = 0; j < num_sample; ++j)
            {
                da.data[i][j] = dz.data[i][j] * d_cost.data[0][i] / num_sample;
            }
        }

        params_to_csv(out_params, w, iter, cost, totaltime, correct);

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
            }
        }

        for(int i = 0; i < w.rows; ++i)
        {
            for (int j = 0; j < w.cols; ++j)
            {
                dw.data[i][j] /= BATCH_SIZE;
                w.data[i][j] -= lr * dw.data[i][j];
            }
        }

        end = clock();
        elapsed = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
        totaltime += elapsed;

        if ( iter % 10 == 0 )
            log_debug("GD = ITER: %d, COST: %lf, CORRECT: %lf", iter, cost, correct);

    }

    fclose(out_params);

    matrix_destroy(&w);
    matrix_destroy(&z);
    matrix_destroy(&a);
    matrix_destroy(&dz);
    matrix_destroy(&da);
    matrix_destroy(&dw);
    matrix_destroy(&d_cost);
}

void adam(matrix x, matrix y, matrix w_, const char* param_name, matrix x_test, matrix y_test)
{
    clock_t start, end;
    double elapsed = 0;
    double totaltime = 0;

    matrix w = { 0 };
    matrix_copy(&w, w_);

    matrix z = { 0 };
    matrix a = { 0 };

    matrix dz = { 0 };
    matrix da = { 0 };

    matrix m = { 0 };
    matrix_init(&m, w.rows, w.cols);
    matrix_zero(m);

    matrix mt = { 0 };
    matrix_init(&mt, w.rows, w.cols);
    matrix_zero(mt);

    matrix v = { 0 };
    matrix_init(&v, w.rows, w.cols);
    matrix_zero(v);

    matrix vt = { 0 };
    matrix_init(&vt, w.rows, w.cols);
    matrix_zero(vt);

    int t = 0;

    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 0.00000001;
    double alpha = 0.01;

    char outdir[64] = { 0 };
    sprintf(outdir, "out/adam_%s.csv", param_name);
    FILE *out_params = fopen(outdir, "w");

    for (int iter = 0; iter < EPOCHS; ++iter)
    {
        matrix_dot(&z, w, x);
        matrix_tanh(&a, z);

        int dataidx = iter % x.rows;

        double d_cost = 0;
        double cost = 0;
        for (int i = 0; i < a.cols; ++i)
        {
            double y_hat = a.data[0][i];
            double expected = y.data[0][i];
            double diff = y_hat - expected;
            cost += pow(diff, 2);
        }
        cost /= a.cols;

        start = clock();

        double correct = check_correct(w, x_test, y_test);

        params_to_csv(out_params, w, iter, cost, totaltime, correct);

        d_cost = 2 * ( a.data[0][dataidx] - y.data[0][dataidx] );

        matrix_tanh_deriv(&dz, z);
        matrix_scale(&da, dz, d_cost);

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
        }

        for (int i = 0; i < w.rows; ++i)
        {
            for (int j = 0; j < w.cols; ++j)
            {
                w.data[i][j] -= alpha * mt.data[i][j] / (pow(vt.data[i][j], 0.5) + epsilon);
            }
        }

        end = clock();
        elapsed = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
        totaltime += elapsed;

        if ( iter % 10 == 0 )
            log_debug("ADAM = ITER: %d, COST: %lf, CORRECT: %lf", iter, cost, correct);
    }

    fclose(out_params);

    matrix_destroy(&w);
    matrix_destroy(&z);
    matrix_destroy(&a);
    matrix_destroy(&dz);
    matrix_destroy(&da);
    matrix_destroy(&m);
    matrix_destroy(&mt);
    matrix_destroy(&v);
    matrix_destroy(&vt);
}

void params_to_csv(FILE *file, matrix w, int iter, double cost, double elapsed, double correct)
{
    fprintf(file, "%d,%lf,%lf,%lf,", iter, cost, elapsed, correct);
    for (int i = 0; i < w.cols - 1; ++i)
        fprintf(file, "%.3lf,", w.data[0][i]);
    fprintf(file, "%.3lf\n", w.data[0][w.cols-1]);
}

double check_correct(matrix w, matrix x_test, matrix y_test)
{
    matrix z = { 0 };
    matrix a = { 0 };

    matrix_dot(&z, w, x_test);
    matrix_tanh(&a, z);

    int correct = 0;
    for (int i = 0; i < y_test.cols; ++i)
    {
        int iscorr = 1;
        for(int j = 0; j < y_test.rows; ++j)
        {
            double predicted = a.data[j][i];
            if ( predicted > 1 || predicted < -1 )
                log_error("error predict: %lf", predicted);

            if ( predicted > 0 )
            {
                if ( y_test.data[0][i] < 0 )
                    iscorr = 0;
            }
            else 
            {
                if ( y_test.data[0][i] >= 0 )
                    iscorr = 0;
            }
        }
        if ( iscorr )
            correct++;
    }

    matrix_destroy(&z);
    matrix_destroy(&a);

    return (double)correct / y_test.cols;
}
