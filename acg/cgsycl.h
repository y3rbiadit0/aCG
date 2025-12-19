/* This file is part of acg.
 *
 * Copyright 2025 Koç University and Simula Research Laboratory
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the “Software”), to deal in the Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * Authors: James D. Trotter <james@simula.no>
 *
 * Last modified: 2025-04-26
 *
 * conjugate gradient (CG) solver using SYCL
 */

#ifndef ACG_CGSYCL_H
#define ACG_CGSYCL_H

#include "acg/config.h"
#include "acg/vector.h"
#include "vector.h"

#ifdef ACG_HAVE_MPI
#include <mpi.h>
#endif

#include <sycl/sycl.hpp>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

struct acgcomm;
struct acghalo;
struct acgsymcsrmatrix;

/**
 * ‘acgsolversycl’ is a data structure for use with a conjugate
 * gradient-based iterative solver.
 */
struct acgsolversycl {
  /* vectors */
  struct acgvector r;
  struct acgvector p;
  struct acgvector t;
  struct acgvector *w, *q, *z;
  struct acgvector *dx;

  /* “halo exchange” communication */
  struct acghalo *halo;
  struct acghaloexchange *haloexchange;

  /* stopping criterion */
  int maxits;
  double diffatol;
  double diffrtol;
  double residualatol;
  double residualrtol;

  /* norms */
  double bnrm2;
  double r0nrm2, rnrm2;
  double x0nrm2, dxnrm2;

  /* SYCL queue */
  sycl::queue *queue;

  /* device-side data (USM pointers) */
  double *d_bnrm2sqr, *d_r0nrm2sqr, *d_rnrm2sqr, *d_rnrm2sqr_prev;
  double *d_pdott, *d_alpha, *d_minus_alpha, *d_beta;
  int *d_niterations, *d_converged;
  double *d_r, *d_p, *d_t, *d_w, *d_q, *d_z;
  acgidx_t *d_rowptr, *d_orowptr;
  acgidx_t *d_colidx, *d_ocolidx;
  double *d_a, *d_oa;

  /* constants */
  double *d_minus_one, *d_one, *d_zero, *d_inf;

  /* solver statistics */
  int nsolves, ntotaliterations, niterations;
  int64_t nflops;
  double tsolve;
  double tgemv, tdot, tnrm2, taxpy, tcopy, tallreduce, thalo;
  int64_t ngemv, ndot, nnrm2, naxpy, ncopy, nallreduce, nhalo;
  int64_t Bgemv, Bdot, Bnrm2, Baxpy, Bcopy, Ballreduce, Bhalo;
  int64_t nhalomsgs;
};

/*
 * memory management
 */

ACG_API void acgsolversycl_free(struct acgsolversycl *cg);

/*
 * initialise a solver
 */

ACG_API int acgsolversycl_init(struct acgsolversycl *cg,
                               const struct acgsymcsrmatrix *A,
                               sycl::queue &queue, const struct acgcomm *comm);

ACG_API int acgsolversycl_solve(struct acgsolversycl *cg,
                                const struct acgsymcsrmatrix *A,
                                const struct acgvector *b, struct acgvector *x,
                                int maxits, double diffatol, double diffrtol,
                                double residualatol, double residualrtol,
                                int warmup);

#ifdef ACG_HAVE_MPI
ACG_API int acgsolversycl_solvempi(
    struct acgsolversycl *cg, const struct acgsymcsrmatrix *A,
    const struct acgvector *b, struct acgvector *x, int maxits, double diffatol,
    double diffrtol, double residualatol, double residualrtol, int warmup,
    struct acgcomm *comm, int tag, int *errcode);
#endif

ACG_API int acgsolversycl_fwrite(FILE *f, const struct acgsolversycl *cg,
                                 int indent);

#ifdef ACG_HAVE_MPI
ACG_API int acgsolversycl_fwritempi(FILE *f, const struct acgsolversycl *cg,
                                    int indent, int verbose, MPI_Comm comm,
                                    int root);
#endif

#ifdef __cplusplus
}
#endif

#endif
