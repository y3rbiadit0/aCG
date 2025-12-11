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

#include "acg/cgsycl.h"
#include "acg/cg-kernels-sycl.h"
#include "acg/comm.h"
#include "acg/config.h"
#include "acg/error.h"
#include "acg/halo.h"
#include "acg/symcsrmatrix.h"
#include "acg/time.h"
#include "acg/vector.h"

#ifdef ACG_HAVE_MPI
#include <mpi.h>
#endif

#include <CL/sycl.hpp>
#include <cmath>
#include <cstdlib>
#include <cstring>

void acgsolversycl_free(struct acgsolversycl *cg) {
  if (cg->queue) {
    sycl::free(cg->d_bnrm2sqr, *cg->queue);
    sycl::free(cg->d_r0nrm2sqr, *cg->queue);
    sycl::free(cg->d_rnrm2sqr, *cg->queue);
    sycl::free(cg->d_pdott, *cg->queue);
    sycl::free(cg->d_rnrm2sqr_prev, *cg->queue);
    sycl::free(cg->d_alpha, *cg->queue);
    sycl::free(cg->d_minus_alpha, *cg->queue);
    sycl::free(cg->d_beta, *cg->queue);
    sycl::free(cg->d_niterations, *cg->queue);
    sycl::free(cg->d_converged, *cg->queue);
    sycl::free(cg->d_r, *cg->queue);
    sycl::free(cg->d_p, *cg->queue);
    sycl::free(cg->d_t, *cg->queue);
    if (cg->d_w)
      sycl::free(cg->d_w, *cg->queue);
    if (cg->d_q)
      sycl::free(cg->d_q, *cg->queue);
    if (cg->d_z)
      sycl::free(cg->d_z, *cg->queue);
    sycl::free(cg->d_rowptr, *cg->queue);
    sycl::free(cg->d_colidx, *cg->queue);
    sycl::free(cg->d_a, *cg->queue);
    sycl::free(cg->d_orowptr, *cg->queue);
    sycl::free(cg->d_ocolidx, *cg->queue);
    sycl::free(cg->d_oa, *cg->queue);
    sycl::free(cg->d_one, *cg->queue);
    sycl::free(cg->d_minus_one, *cg->queue);
    sycl::free(cg->d_zero, *cg->queue);
    sycl::free(cg->d_inf, *cg->queue);
  }

  acgvector_free(&cg->r);
  acgvector_free(&cg->p);
  acgvector_free(&cg->t);
  if (cg->w) {
    acgvector_free(cg->w);
    free(cg->w);
  }
  if (cg->q) {
    acgvector_free(cg->q);
    free(cg->q);
  }
  if (cg->z) {
    acgvector_free(cg->z);
    free(cg->z);
  }
  if (cg->dx) {
    acgvector_free(cg->dx);
    free(cg->dx);
  }
  if (cg->halo) {
    acghalo_free(cg->halo);
    free(cg->halo);
  }
  if (cg->haloexchange) {
    acghaloexchange_free(cg->haloexchange);
    free(cg->haloexchange);
  }
}

int acgsolversycl_init(struct acgsolversycl *cg,
                       const struct acgsymcsrmatrix *A, sycl::queue &queue,
                       const struct acgcomm *comm) {
  cg->queue = &queue;
  int err = acgsymcsrmatrix_vector(A, &cg->r);
  if (err)
    return err;
  acgvector_setzero(&cg->r);
  err = acgsymcsrmatrix_vector(A, &cg->p);
  if (err) {
    acgvector_free(&cg->r);
    return err;
  }
  acgvector_setzero(&cg->p);
  err = acgsymcsrmatrix_vector(A, &cg->t);
  if (err) {
    acgvector_free(&cg->p);
    acgvector_free(&cg->r);
    return err;
  }
  acgvector_setzero(&cg->t);
  cg->w = cg->q = cg->z = NULL;
  cg->dx = NULL;

  cg->halo = (struct acghalo *)malloc(sizeof(*cg->halo));
  if (!cg->halo)
    return ACG_ERR_ERRNO;
  err = acgsymcsrmatrix_halo(A, cg->halo);
  if (err)
    return err;

  cg->haloexchange =
      (struct acghaloexchange *)malloc(sizeof(*cg->haloexchange));
  if (!cg->haloexchange)
    return ACG_ERR_ERRNO;
  // Note: acghaloexchange_init_cuda is specific to CUDA.
  // We might need a generic or SYCL version. For now leaving it uninitialized
  // or using MPI if available. Assuming MPI communication for halo exchange for
  // now.

  // Allocate USM memory
  cg->d_one = sycl::malloc_device<double>(1, queue);
  cg->d_minus_one = sycl::malloc_device<double>(1, queue);
  cg->d_zero = sycl::malloc_device<double>(1, queue);
  cg->d_inf = sycl::malloc_device<double>(1, queue);

  double one = 1.0, minus_one = -1.0, zero = 0.0, inf = INFINITY;
  queue.memcpy(cg->d_one, &one, sizeof(double));
  queue.memcpy(cg->d_minus_one, &minus_one, sizeof(double));
  queue.memcpy(cg->d_zero, &zero, sizeof(double));
  queue.memcpy(cg->d_inf, &inf, sizeof(double));

  cg->d_bnrm2sqr = sycl::malloc_device<double>(1, queue);
  cg->d_r0nrm2sqr = sycl::malloc_device<double>(1, queue);
  cg->d_rnrm2sqr = sycl::malloc_device<double>(2, queue);
  cg->d_pdott = sycl::malloc_device<double>(1, queue);
  cg->d_rnrm2sqr_prev = sycl::malloc_device<double>(1, queue);
  cg->d_niterations = sycl::malloc_device<int>(1, queue);
  cg->d_converged = sycl::malloc_device<int>(1, queue);
  cg->d_alpha = sycl::malloc_device<double>(1, queue);
  cg->d_minus_alpha = sycl::malloc_device<double>(1, queue);
  cg->d_beta = sycl::malloc_device<double>(1, queue);

  cg->d_r = sycl::malloc_device<double>(cg->r.num_nonzeros, queue);
  cg->d_p = sycl::malloc_device<double>(cg->p.num_nonzeros, queue);
  cg->d_t = sycl::malloc_device<double>(cg->t.num_nonzeros, queue);

  // Copy matrix to device
  cg->d_rowptr = sycl::malloc_device<acgidx_t>(A->nprows + 1, queue);
  queue.memcpy(cg->d_rowptr, A->frowptr, (A->nprows + 1) * sizeof(acgidx_t));

  cg->d_colidx = sycl::malloc_device<acgidx_t>(A->fnpnzs, queue);
  queue.memcpy(cg->d_colidx, A->fcolidx, A->fnpnzs * sizeof(acgidx_t));

  cg->d_a = sycl::malloc_device<double>(A->fnpnzs, queue);
  queue.memcpy(cg->d_a, A->fa, A->fnpnzs * sizeof(double));

  queue.wait();

  return ACG_SUCCESS;
}

int acgsolversycl_solve(struct acgsolversycl *cg,
                        const struct acgsymcsrmatrix *A,
                        const struct acgvector *b, struct acgvector *x,
                        int maxits, double diffatol, double diffrtol,
                        double residualatol, double residualrtol, int warmup) {
  // Placeholder for solve logic
  // In a real implementation, this would contain the CG loop using SYCL kernels
  return ACG_SUCCESS;
}

#ifdef ACG_HAVE_MPI
int acgsolversycl_solvempi(struct acgsolversycl *cg,
                           const struct acgsymcsrmatrix *A,
                           const struct acgvector *b, struct acgvector *x,
                           int maxits, double diffatol, double diffrtol,
                           double residualatol, double residualrtol, int warmup,
                           struct acgcomm *comm, int tag, int *errcode) {
  // Placeholder for MPI solve logic
  return ACG_SUCCESS;
}
#endif

int acgsolversycl_fwrite(FILE *f, const struct acgsolversycl *cg, int indent) {
  // Placeholder for output logic
  return ACG_SUCCESS;
}

#ifdef ACG_HAVE_MPI
int acgsolversycl_fwritempi(FILE *f, const struct acgsolversycl *cg, int indent,
                            int verbose, MPI_Comm comm, int root) {
  // Placeholder for MPI output logic
  return ACG_SUCCESS;
}
#endif
