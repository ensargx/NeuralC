#include "matrix.h"

#include "util/logger.h"

void matrix_init(matrix *pMatrix, int rows, int cols) 
{
    if (rows < 0 || cols < 0)
    {
        log_error("[%s]: Wrong parameters: rows || cols", __FUNCTION__);
        return;
    }

}
