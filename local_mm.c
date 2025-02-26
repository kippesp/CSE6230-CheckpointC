/**
 *  \file local_mm.c
 *  \brief Matrix Multiply file for Proj1
 *  \author Kent Czechowski <kentcz@gatech...>, Rich Vuduc <richie@gatech...>
 */

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

#if defined(USE_OPEN_MP) || defined(USE_BLOCKING)
# include <omp.h>
#endif

#ifdef USE_MKL
# include <mkl.h>
#endif

#define MIN(a, b)   ((a < b) ? a : b)

static void print_matrix(int rows, int cols, const double *mat) {

  int r, c;

  /* Iterate over the rows of the matrix */
  for (r = 0; r < rows; r++) {
    /* Iterate over the columns of the matrix */
    for (c = 0; c < cols; c++) {
      int index = (c * rows) + r;
      fprintf(stderr, "%2.0lf ", mat[index]);
    } /* c */
    fprintf(stderr, "\n");
  } /* r */
}

/**
 *
 *  Local Matrix Multiply
 *   Computes C = alpha * A * B + beta * C
 *
 *
 *  Similar to the DGEMM routine in BLAS
 *
 *
 *  alpha and beta are double-precision scalars
 *
 *  A, B, and C are matrices of double-precision elements
 *  stored in column-major format
 *
 *  The output is stored in C
 *  A and B are not modified during computation
 *
 *
 *  m - number of rows of matrix A and rows of C
 *  n - number of columns of matrix B and columns of C
 *  k - number of columns of matrix A and rows of B
 *
 *  lda, ldb, and ldc specifies the size of the first dimension of the matrices
 *
 **/
void local_mm(const int m, const int n, const int k, const double alpha,
    const double *A, const int lda, const double *B, const int ldb,
    const double beta, double *C, const int ldc) {

  int row, col;

  /* Verify the sizes of lda, ladb, and ldc */
  assert(lda >= m);
  assert(ldb >= k);
  assert(ldc >= m);

#ifdef USE_MKL
  const char N = 'N';
  dgemm(&N, &N, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
#else
# ifdef USE_BLOCKING

  /*
   * Z = 256 KB = 256 * 1024 = 262144
   * b = sqrt(X) = 512
   * sizeof(double) = 8
   *
   * n = 1024
   * m = n^3 / b = 2097152
   *
   *
   *
   */


#pragma omp parallel private(col, row) shared(C)
  {
  int tid;
  int nthreads;

  tid = omp_get_thread_num();
  nthreads = omp_get_num_threads();


  if (tid == 1)
  {
    fprintf(stderr, "nthreads=%i, tid*n/nthreads=%i, tid*m/nthreads=%i\n",
        nthreads, tid*n/nthreads, tid*m/nthreads);
    fprintf(stderr, "(tid+1)*n/nthreads=%i, (tid+1)*m/nthreads=%i\n",
        (tid+1)*n/nthreads, (tid+1)*m/nthreads);

    //fprintf(stderr, "MATRIX A=\n");
    //print_matrix(m, n, A);

    //fprintf(stderr, "\bMATRIX B=\n");
    //print_matrix(m, n, B);
  }

  /* Iterate over the columns of C */
  for (col = 0; col < n; col++) {

    /* Spread the computations among the CPUs; the last CPU may get fewer rows. */
    int row_min = tid * ((float)m/nthreads + 0.5);
    int row_max = MIN((tid+1) * ((float)m/nthreads + 0.5), m);

    /* Iterate over the rows of C */
    for (row = row_min; row < row_max; row++) {

      int k_iter;
      double dotprod = 0.0; /* Accumulates the sum of the dot-product */

      /* Iterate over column of A, row of B */
      for (k_iter = 0; k_iter < k; k_iter++) {
        int a_index, b_index;
        a_index = (k_iter * lda) + row; /* Compute index of A element */
        b_index = (col * ldb) + k_iter; /* Compute index of B element */
        dotprod += A[a_index] * B[b_index]; /* Compute product of A and B */
      } /* k_iter */

      int c_index = (col * ldc) + row;
      C[c_index] = (alpha * dotprod) + (beta * C[c_index]);
    } /* row */
  } /* col */

  }

# else /* OPEN_MP */

#pragma omp parallel for private(col, row)
  /* Iterate over the columns of C */
  for (col = 0; col < n; col++) {

    /* Iterate over the rows of C */
    for (row = 0; row < m; row++) {

      int k_iter;
      double dotprod = 0.0; /* Accumulates the sum of the dot-product */

      /* Iterate over column of A, row of B */
      for (k_iter = 0; k_iter < k; k_iter++) {
        int a_index, b_index;
        a_index = (k_iter * lda) + row; /* Compute index of A element */
        b_index = (col * ldb) + k_iter; /* Compute index of B element */
        dotprod += A[a_index] * B[b_index]; /* Compute product of A and B */
      } /* k_iter */

      int c_index = (col * ldc) + row;
      C[c_index] = (alpha * dotprod) + (beta * C[c_index]);
    } /* row */
  } /* col */
# endif /* USE_BLOCKING, OPEN_MP */
#endif /* USE_MKL */

}
