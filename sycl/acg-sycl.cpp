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
 * Authors:
 *  James D. Trotter <james@simula.no>
 *  Sinan Ekmekçibaşı <sekmekcibasi23@ku.edu.tr>
 *
 * Last modified: 2025-04-26
 *
 * Example application for multi-GPU solvers using the conjugate
 * gradient (CG) method.
 *
 */

#include "acg/cgpetsc.h"
#include "acg/cgsycl.h"
#include "acg/comm.h"
#include "acg/config.h"
#include "acg/error.h"
extern "C"
{
#include "../acg/fmtspec.h"
}
#include "acg/graph.h"
#include "acg/halo.h"
#include "acg/mtxfile.h"
#include "acg/symcsrmatrix.h"
#include "acg/time.h"
#include "acg/vector.h"

#ifdef ACG_HAVE_OPENMP
#include <omp.h>
#endif
#ifdef ACG_HAVE_PETSC
#include <petsc.h>
#endif
#ifdef ACG_HAVE_MPI
#include <mpi.h>
#endif
#ifdef ACG_HAVE_METIS
#include <metis.h>
#endif

#include <sycl/sycl.hpp>

#include <float.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#include <errno.h>
#include <sched.h>
#include <math.h>
#include "acg-sycl.h"

const char *program_name = "acg-sycl";
const char *program_version = "0.9.4";
const char *program_copyright =
    "Copyright (C) 2025 Simula Research Laboratory, Koç University";
const char *program_license =
    "Copyright 2025 Koç University and Simula Research Laboratory\n"
    "\n"
    "Permission is hereby granted, free of charge, to any person\n"
    "obtaining a copy of this software and associated documentation\n"
    "files (the “Software”), to deal in the Software without\n"
    "restriction, including without limitation the rights to use, copy,\n"
    "modify, merge, publish, distribute, sublicense, and/or sell copies\n"
    "of the Software, and to permit persons to whom the Software is\n"
    "furnished to do so, subject to the following conditions:\n"
    "\n"
    "The above copyright notice and this permission notice shall be\n"
    "included in all copies or substantial portions of the Software.\n"
    "\n"
    "THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND,\n"
    "EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF\n"
    "MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND\n"
    "NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS\n"
    "BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN\n"
    "ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN\n"
    "CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\n"
    "SOFTWARE.\n";

/*
 * solver types
 */

enum acgsolvertype
{
  acgsolver_acg,   /* native CG */
  acgsolver_petsc, /* PETSc CG */
};

const char *acgsolvertypestr(enum acgsolvertype solvertype)
{
  if (solvertype == acgsolver_acg)
  {
    return "acg";
  }
  else if (solvertype == acgsolver_petsc)
  {
    return "petsc";
  }
  else
  {
    return "unknown";
  }
}

/*
 * parsing numbers (simplified for brevity, assuming standard parsing functions
 * exist or copying from cuda version if needed)
 */
// ... (omitting parsing functions for brevity, assuming they are available or I
// can copy them if needed. For now I will include the necessary headers and
// assume standard C functions or copy the helpers if they are not in a shared
// header) Actually, the parsing functions were static in acg-cuda.c, so I
// should copy them.

static int parse_long_long_int(const char *s, char **outendptr, int base,
                               long long int *out_number, int64_t *bytes_read)
{
  errno = 0;
  char *endptr;
  long long int number = strtoll(s, &endptr, base);
  if ((errno == ERANGE && (number == LLONG_MAX || number == LLONG_MIN)) ||
      (errno != 0 && number == 0))
    return errno;
  if (outendptr)
    *outendptr = endptr;
  if (bytes_read)
    *bytes_read += endptr - s;
  *out_number = number;
  return 0;
}

int parse_int(int *x, const char *s, char **endptr, int64_t *bytes_read)
{
  long long int y;
  int err = parse_long_long_int(s, endptr, 10, &y, bytes_read);
  if (err)
    return err;
  if (y < INT_MIN || y > INT_MAX)
    return ERANGE;
  *x = y;
  return 0;
}

int parse_int32_t(int32_t *x, const char *s, char **endptr,
                  int64_t *bytes_read)
{
  long long int y;
  int err = parse_long_long_int(s, endptr, 10, &y, bytes_read);
  if (err)
    return err;
  if (y < INT32_MIN || y > INT32_MAX)
    return ERANGE;
  *x = y;
  return 0;
}

int parse_int64_t(int64_t *x, const char *s, char **endptr,
                  int64_t *bytes_read)
{
  long long int y;
  int err = parse_long_long_int(s, endptr, 10, &y, bytes_read);
  if (err)
    return err;
  if (y < INT64_MIN || y > INT64_MAX)
    return ERANGE;
  *x = y;
  return 0;
}

int parse_double(double *x, const char *s, char **outendptr,
                 int64_t *bytes_read)
{
  errno = 0;
  char *endptr;
  *x = strtod(s, &endptr);
  if ((errno == ERANGE && (*x == HUGE_VAL || *x == -HUGE_VAL)) ||
      (errno != 0 && x == 0))
  {
    return errno;
  }
  if (outendptr)
    *outendptr = endptr;
  if (bytes_read)
    *bytes_read += endptr - s;
  return 0;
}

#ifndef ACG_IDX_SIZE
#define parse_acgidx_t parse_int
#elif ACG_IDX_SIZE == 32
#define parse_acgidx_t parse_int32_t
#elif ACG_IDX_SIZE == 64
#define parse_acgidx_t parse_int64_t
#endif

/*
 * program options and help text
 */

/**
 * ‘program_options_print_usage()’ prints a usage text.
 */
static void program_options_print_usage(
    FILE *f)
{
  fprintf(f, "Usage: %s [OPTION..] A [b] [x0]\n", program_name);
}

/**
 * ‘program_options_print_help()’ prints a help text.
 */
static void program_options_print_help(
    FILE *f)
{
  program_options_print_usage(f);
  fprintf(f, "\n");
  fprintf(f, " Solve a linear system of equations ‘Ax=b’ using the conjugate gradient (CG)\n");
  fprintf(f, " method for a matrix ‘A’ and right-hand side vector ‘b’.\n");
  fprintf(f, "\n");
  fprintf(f, " Positional arguments:\n");
  fprintf(f, "  A    path to Matrix Market file for a matrix A\n");
  fprintf(f, "  b    optional path to Matrix Market file for a right-hand side vector b\n");
  fprintf(f, "  x0   optional path to Matrix Market file for an initial guess x0\n");
#ifdef ACG_HAVE_LIBZ
  fprintf(f, "\n");
  fprintf(f, " Input options:\n");
  fprintf(f, "  -z, --gzip, --gunzip, --ungzip    filter files through gzip\n");
#endif
  fprintf(f, "  --binary              read Matrix Market files in binary format\n");
  fprintf(f, "\n");
  fprintf(f, " Partitioning options:\n");
  fprintf(f, "  --partition=FILE      read partition vector from Matrix Market file.\n");
  fprintf(f, "  --binary-partition    read partition vector in binary format\n");
  fprintf(f, "  --seed=N              random number seed. [0]\n");
  fprintf(f, "\n");
  fprintf(f, " Solver options:\n");
  fprintf(f, "  --solver TYPE         acg, acg-pipelined, acg-device, acg-pipelined-device or petsc. [acg]\n");
  fprintf(f, "  --max-iterations N    maximum number of iterations. [100]\n");
  fprintf(f, "  --diff-atol TOL       stopping criterion for difference in solution iterates, ‖xₖ₊₁-xₖ‖ < TOL. [0]\n");
  fprintf(f, "  --diff-rtol TOL       stopping criterion for relative difference in solution iterates, ‖xₖ₊₁-xₖ‖/‖x₀‖ < TOL. [0]\n");
  fprintf(f, "  --residual-atol TOL   stopping criterion for residual norm, ‖b-Ax‖ < TOL. [0]\n");
  fprintf(f, "  --residual-rtol TOL   stopping criterion for relative residual norm, ‖b-Ax‖/‖b‖ < TOL. [1e-9]\n");
  fprintf(f, "  --epsilon TOL         add TOL to the diagonal of A. [0]\n");
  fprintf(f, "  --warmup N            perform N warmup iterations. [10]\n");
  fprintf(f, "\n");
  fprintf(f, " Communication library options:\n");
  fprintf(f, "  --comm TYPE           none, mpi, nccl or nvshmem. [mpi]\n");
  fprintf(f, "\n");
  fprintf(f, " Solver verification options:\n");
  fprintf(f, "  --manufactured-solution  Use a manufactured solution and right-hand side.\n");
  fprintf(f, "\n");
  fprintf(f, " Output options:\n");
  /* fprintf(f, "  --repeat=N           repeat solver N times\n"); */
  fprintf(f, "  --numfmt FMT         Format string for outputting numerical values.\n");
  fprintf(f, "                       The format specifiers '%%e', '%%E', '%%f', '%%F',\n");
  fprintf(f, "                       '%%g' or '%%G' may be used. Flags, field width and\n");
  fprintf(f, "                       precision may also be specified, e.g., \"%%+3.1f\".\n");
  fprintf(f, "  --output-comm-matrix print communication matrix to standard output\n");
  fprintf(f, "\n");
  fprintf(f, "  -v, --verbose        be more verbose\n");
  fprintf(f, "  -q, --quiet          suppress output\n");
  fprintf(f, "\n");
  fprintf(f, " Other options:\n");
  fprintf(f, "  -h, --help           display this help and exit\n");
  fprintf(f, "  --version            display version information and exit\n");
  fprintf(f, "\n");
  fprintf(f, "Report bugs to: <james@simula.no>\n");
}

/**
 * ‘program_options_print_version()’ prints version information.
 */
static void program_options_print_version(
    FILE *f)
{
  fprintf(f, "%s %s\n", program_name, program_version);
  fprintf(f, "32/64-bit integers: %ld-bit\n", sizeof(acgidx_t) * CHAR_BIT);
#ifdef ACG_ENABLE_PROFILING
  fprintf(f, "profiling: enabled\n");
#else
  fprintf(f, "profiling: disabled\n");
#endif
#ifdef ACG_HAVE_MPI
  char mpistr[MPI_MAX_LIBRARY_VERSION_STRING] = "";
  int len;
  MPI_Get_library_version(mpistr, &len);
  fprintf(f, "MPI: %d.%d (%s)\n", MPI_VERSION, MPI_SUBVERSION, mpistr);
#else
  fprintf(f, "MPI: no\n");
#endif
#ifdef ACG_HAVE_NCCL
  int ncclversion = 0;
  ncclGetVersion(&ncclversion);
  fprintf(f, "nccl: %d\n", ncclversion);
#else
  fprintf(f, "nccl: no\n");
#endif
#ifdef ACG_HAVE_NVSHMEM
  int nvshmemmajor, nvshmemminor, nvshmempatch;
  acg_nvshmemx_vendor_get_version_info(&nvshmemmajor, &nvshmemminor, &nvshmempatch);
  fprintf(f, "NVSHMEM: %d.%d.%d\n", nvshmemmajor, nvshmemminor, nvshmempatch);
#else
  fprintf(f, "NVSHMEM: no\n");
#endif
#ifdef ACG_HAVE_LIBZ
  fprintf(f, "zlib: " ZLIB_VERSION "\n");
#else
  fprintf(f, "zlib: no\n");
#endif
#ifdef ACG_HAVE_METIS
  fprintf(f, "metis: %d.%d.%d (%d-bit index, %d-bit real)\n",
          METIS_VER_MAJOR, METIS_VER_MINOR, METIS_VER_SUBMINOR,
          IDXTYPEWIDTH, REALTYPEWIDTH);
#else
  fprintf(f, "metis: no\n");
#endif
#ifdef ACG_HAVE_PETSC
  fprintf(f, "PETSc: %d.%d.%d\n", PETSC_VERSION_MAJOR, PETSC_VERSION_MINOR, PETSC_VERSION_SUBMINOR);
#else
  fprintf(f, "PETSc: no\n");
#endif
  fprintf(f, "\n");
  fprintf(f, "%s\n", program_copyright);
  fprintf(f, "%s\n", program_license);
}

struct program_options
{
  /* input options */
  char *Apath;
  char *bpath;
  char *x0path;
  int gzip;
  int binary;
  int rowpartbinary;

  /* partitioning options */
  char *rowpartspath;
  acgidx_t seed;

  /* linear solver options */
  enum acgsolvertype solvertype;
  double diffatol, diffrtol;
  double residualatol, residualrtol;
  int maxits;
  double epsilon;
  int warmup;

  /* communication library options */
  enum acgcommtype commtype;

  /* solver verification options */
  int manufactured_solution;

  /* output options */
  char *numfmt;
  int output_comm_matrix;
  int verbose;
  bool quiet;

  /* other options */
  bool help;
  bool version;
};

static int program_options_init(struct program_options *args)
{
  args->Apath = NULL;
  args->bpath = NULL;
  args->x0path = NULL;
  args->gzip = 0;
  args->binary = 0;
  args->rowpartbinary = 0;

  args->rowpartspath = NULL;
  args->seed = 0;

  args->solvertype = acgsolver_acg;
  args->diffatol = args->diffrtol = 0;
  args->residualatol = 0;
  args->residualrtol = 1e-9;
  args->maxits = 100;
  args->epsilon = 0;
  args->warmup = 10;

  args->commtype = acgcomm_mpi;

  args->manufactured_solution = 0;

  args->numfmt = NULL;
  args->output_comm_matrix = 0;
  args->verbose = 0;
  args->quiet = false;

  args->help = false;
  args->version = false;
  return 0;
}

static void program_options_free(struct program_options *args)
{
  if (args->Apath)
    free(args->Apath);
  if (args->bpath)
    free(args->bpath);
  if (args->x0path)
    free(args->x0path);
  if (args->rowpartspath)
    free(args->rowpartspath);
  if (args->numfmt)
    free(args->numfmt);
}

static int parse_program_options(int argc, char **argv,
                                 struct program_options *args, int *nargs)
{
  *nargs = 0;
  (*nargs)++;
  argv++;

  int num_positional_arguments_consumed = 0;
  while (*nargs < argc)
  {

#ifdef ACG_HAVE_LIBZ
    if (strcmp(argv[0], "-z") == 0 || strcmp(argv[0], "--gzip") == 0 ||
        strcmp(argv[0], "--gunzip") == 0 || strcmp(argv[0], "--ungzip") == 0)
    {
      args->gzip = 1;
      (*nargs)++;
      argv++;
      continue;
    }
#endif
    if (strcmp(argv[0], "--binary") == 0)
    {
      args->binary = 1;
      (*nargs)++;
      argv++;
      continue;
    }
    if (strcmp(argv[0], "--binary-partition") == 0)
    {
      args->rowpartbinary = 1;
      (*nargs)++;
      argv++;
      continue;
    }

    if (strstr(argv[0], "--partition") == argv[0])
    {
      int n = strlen("--partition");
      const char *s = &argv[0][n];
      if (*s == '=')
      {
        s++;
      }
      else if (*s == '\0' && argc - *nargs > 1)
      {
        (*nargs)++;
        argv++;
        s = argv[0];
      }
      else
      {
        return EINVAL;
      }
      args->rowpartspath = strdup(s);
      if (!args->rowpartspath)
        return EINVAL;
      (*nargs)++;
      argv++;
      continue;
    }
    if (strstr(argv[0], "--seed") == argv[0])
    {
      int n = strlen("--seed");
      const char *s = &argv[0][n];
      if (*s == '=')
      {
        s++;
      }
      else if (*s == '\0' && argc - *nargs > 1)
      {
        (*nargs)++;
        argv++;
        s = argv[0];
      }
      else
      {
        return EINVAL;
      }
      char *endptr;
      if (parse_acgidx_t(&args->seed, s, &endptr, NULL))
        return EINVAL;
      if (*endptr != '\0')
        return EINVAL;
      (*nargs)++;
      argv++;
      continue;
    }

    if (strstr(argv[0], "--solver") == argv[0])
    {
      int n = strlen("--solver");
      const char *s = &argv[0][n];
      if (*s == '=')
      {
        s++;
      }
      else if (*s == '\0' && argc - *nargs > 1)
      {
        (*nargs)++;
        argv++;
        s = argv[0];
      }
      else
      {
        return EINVAL;
      }
      if (strcasecmp(s, "acg") == 0)
      {
        args->solvertype = acgsolver_acg;
      }
      else if (strcasecmp(s, "petsc") == 0)
      {
        args->solvertype = acgsolver_petsc;
      }
      else
      {
        return EINVAL;
      }
      (*nargs)++;
      argv++;
      continue;
    }
    if (strstr(argv[0], "--diff-atol") == argv[0])
    {
      int n = strlen("--diff-atol");
      const char *s = &argv[0][n];
      if (*s == '=')
      {
        s++;
      }
      else if (*s == '\0' && argc - *nargs > 1)
      {
        (*nargs)++;
        argv++;
        s = argv[0];
      }
      else
      {
        return EINVAL;
      }
      char *endptr;
      if (parse_double(&args->diffatol, s, &endptr, NULL))
        return EINVAL;
      if (*endptr != '\0')
        return EINVAL;
      (*nargs)++;
      argv++;
      continue;
    }
    if (strstr(argv[0], "--diff-rtol") == argv[0])
    {
      int n = strlen("--diff-rtol");
      const char *s = &argv[0][n];
      if (*s == '=')
      {
        s++;
      }
      else if (*s == '\0' && argc - *nargs > 1)
      {
        (*nargs)++;
        argv++;
        s = argv[0];
      }
      else
      {
        return EINVAL;
      }
      char *endptr;
      if (parse_double(&args->diffrtol, s, &endptr, NULL))
        return EINVAL;
      if (*endptr != '\0')
        return EINVAL;
      (*nargs)++;
      argv++;
      continue;
    }
    if (strstr(argv[0], "--residual-atol") == argv[0])
    {
      int n = strlen("--residual-atol");
      const char *s = &argv[0][n];
      if (*s == '=')
      {
        s++;
      }
      else if (*s == '\0' && argc - *nargs > 1)
      {
        (*nargs)++;
        argv++;
        s = argv[0];
      }
      else
      {
        return EINVAL;
      }
      char *endptr;
      if (parse_double(&args->residualatol, s, &endptr, NULL))
        return EINVAL;
      if (*endptr != '\0')
        return EINVAL;
      (*nargs)++;
      argv++;
      continue;
    }
    if (strstr(argv[0], "--residual-rtol") == argv[0])
    {
      int n = strlen("--residual-rtol");
      const char *s = &argv[0][n];
      if (*s == '=')
      {
        s++;
      }
      else if (*s == '\0' && argc - *nargs > 1)
      {
        (*nargs)++;
        argv++;
        s = argv[0];
      }
      else
      {
        return EINVAL;
      }
      char *endptr;
      if (parse_double(&args->residualrtol, s, &endptr, NULL))
        return EINVAL;
      if (*endptr != '\0')
        return EINVAL;
      (*nargs)++;
      argv++;
      continue;
    }
    if (strstr(argv[0], "--max-iterations") == argv[0])
    {
      int n = strlen("--max-iterations");
      const char *s = &argv[0][n];
      if (*s == '=')
      {
        s++;
      }
      else if (*s == '\0' && argc - *nargs > 1)
      {
        (*nargs)++;
        argv++;
        s = argv[0];
      }
      else
      {
        return EINVAL;
      }
      char *endptr;
      if (parse_int(&args->maxits, s, &endptr, NULL))
        return EINVAL;
      if (*endptr != '\0')
        return EINVAL;
      (*nargs)++;
      argv++;
      continue;
    }
    if (strstr(argv[0], "--epsilon") == argv[0])
    {
      int n = strlen("--epsilon");
      const char *s = &argv[0][n];
      if (*s == '=')
      {
        s++;
      }
      else if (*s == '\0' && argc - *nargs > 1)
      {
        (*nargs)++;
        argv++;
        s = argv[0];
      }
      else
      {
        return EINVAL;
      }
      char *endptr;
      if (parse_double(&args->epsilon, s, &endptr, NULL))
        return EINVAL;
      if (*endptr != '\0')
        return EINVAL;
      (*nargs)++;
      argv++;
      continue;
    }
    if (strstr(argv[0], "--warmup") == argv[0])
    {
      int n = strlen("--warmup");
      const char *s = &argv[0][n];
      if (*s == '=')
      {
        s++;
      }
      else if (*s == '\0' && argc - *nargs > 1)
      {
        (*nargs)++;
        argv++;
        s = argv[0];
      }
      else
      {
        return EINVAL;
      }
      char *endptr;
      if (parse_int(&args->warmup, s, &endptr, NULL))
        return EINVAL;
      if (*endptr != '\0')
        return EINVAL;
      (*nargs)++;
      argv++;
      continue;
    }

    if (strstr(argv[0], "--comm") == argv[0])
    {
      int n = strlen("--comm");
      const char *s = &argv[0][n];
      if (*s == '=')
      {
        s++;
      }
      else if (*s == '\0' && argc - *nargs > 1)
      {
        (*nargs)++;
        argv++;
        s = argv[0];
      }
      else
      {
        return EINVAL;
      }
      if (strcasecmp(s, "none") == 0)
      {
        args->commtype = acgcomm_null;
      }
      else if (strcasecmp(s, "mpi") == 0)
      {
        args->commtype = acgcomm_mpi;
      }
      else
      {
        return EINVAL;
      }
      (*nargs)++;
      argv++;
      continue;
    }

    if (strcmp(argv[0], "--manufactured-solution") == 0)
    {
      args->manufactured_solution = 1;
      (*nargs)++;
      argv++;
      continue;
    }
    else if (strcmp(argv[0], "--no-manufactured-solution") == 0)
    {
      args->manufactured_solution = 0;
      (*nargs)++;
      argv++;
      continue;
    }

    if (strstr(argv[0], "--numfmt") == argv[0])
    {
      int n = strlen("--numfmt");
      const char *s = &argv[0][n];
      if (*s == '=')
      {
        s++;
      }
      else if (*s == '\0' && argc - *nargs > 1)
      {
        (*nargs)++;
        argv++;
        s = argv[0];
      }
      else
      {
        return EINVAL;
      }
      struct fmtspec spec;
      if (fmtspec_parse(&spec, s, NULL))
      {
        return EINVAL;
      }
      args->numfmt = strdup(s);
      if (!args->numfmt)
      {
        free(args->numfmt);
        return EINVAL;
      }
      (*nargs)++;
      argv++;
      continue;
    }
    if (strcmp(argv[0], "--output-comm-matrix") == 0)
    {
      args->output_comm_matrix = 1;
      (*nargs)++;
      argv++;
      continue;
    }
    else if (strcmp(argv[0], "--no-output-comm-matrix") == 0)
    {
      args->output_comm_matrix = 0;
      (*nargs)++;
      argv++;
      continue;
    }

    if (strcmp(argv[0], "-q") == 0 || strcmp(argv[0], "--quiet") == 0)
    {
      args->quiet = true;
      (*nargs)++;
      argv++;
      continue;
    }
    if (strcmp(argv[0], "-v") == 0 || strcmp(argv[0], "--verbose") == 0)
    {
      args->verbose++;
      (*nargs)++;
      argv++;
      continue;
    }
    if (strcmp(argv[0], "-h") == 0 || strcmp(argv[0], "--help") == 0)
    {
      args->help = true;
      (*nargs)++;
      argv++;
      return 0;
    }
    if (strcmp(argv[0], "--version") == 0)
    {
      args->version = true;
      (*nargs)++;
      argv++;
      return 0;
    }

    if (strcmp(argv[0], "--") == 0)
    {
      (*nargs)++;
      argv++;
      break;
    }

    if (argv[0][0] == '-')
    {
      return EINVAL;
    }

    if (num_positional_arguments_consumed == 0)
    {
      args->Apath = strdup(argv[0]);
      if (!args->Apath)
        return EINVAL;
      (*nargs)++;
      argv++;
      num_positional_arguments_consumed++;
      continue;
    }
    if (num_positional_arguments_consumed == 1)
    {
      args->bpath = strdup(argv[0]);
      if (!args->bpath)
        return EINVAL;
      (*nargs)++;
      argv++;
      num_positional_arguments_consumed++;
      continue;
    }
    if (num_positional_arguments_consumed == 2)
    {
      args->x0path = strdup(argv[0]);
      if (!args->x0path)
        return EINVAL;
      (*nargs)++;
      argv++;
      num_positional_arguments_consumed++;
      continue;
    }

    break;
  }

  return 0;
}

int main(int argc, char **argv)
{
  int err = ACG_SUCCESS, errcode = 0;
  bool errexit = false;
  acgtime_t t0, t1;

  /* 1a) initialise MPI */

  const MPI_Comm mpicomm = MPI_COMM_WORLD;
  int commsize, rank;
  const int root = 0;
  int mpierrcode = 0;
  char mpierrstr[MPI_MAX_ERROR_STRING];
  int mpierrstrlen;
  int threadlevel;
  mpierrcode = MPI_Init_thread(
      &argc, &argv, MPI_THREAD_FUNNELED, &threadlevel);
  if (mpierrcode)
  {
    MPI_Error_string(mpierrcode, mpierrstr, &mpierrstrlen);
    fprintf(stderr, "%s: MPI_Init_thread failed with %s\n",
            program_invocation_short_name, mpierrstr);
    MPI_Abort(mpicomm, EXIT_FAILURE);
  }
  mpierrcode = MPI_Comm_size(mpicomm, &commsize);
  if (mpierrcode)
  {
    MPI_Error_string(mpierrcode, mpierrstr, &mpierrstrlen);
    fprintf(stderr, "%s: MPI_Comm_size failed with %s\n",
            program_invocation_short_name, mpierrstr);
    MPI_Abort(mpicomm, EXIT_FAILURE);
  }
  mpierrcode = MPI_Comm_rank(mpicomm, &rank);
  if (mpierrcode)
  {
    MPI_Error_string(mpierrcode, mpierrstr, &mpierrstrlen);
    fprintf(stderr, "%s: MPI_Comm_rank failed with %s\n",
            program_invocation_short_name, mpierrstr);
    MPI_Abort(mpicomm, EXIT_FAILURE);
  }
  int processornamelen = 0;
  char processorname[MPI_MAX_PROCESSOR_NAME + 1];
  MPI_Get_processor_name(processorname, &processornamelen);
  processorname[MPI_MAX_PROCESSOR_NAME] = '\0';

  if (rank == root)
  {
    char mpiversionstr[MPI_MAX_LIBRARY_VERSION_STRING];
    int len;
    mpierrcode = MPI_Get_library_version(mpiversionstr, &len);
    if (mpierrcode)
    {
      MPI_Error_string(mpierrcode, mpierrstr, &mpierrstrlen);
      fprintf(stderr, "%s: MPI_Query_thread failed with %s\n",
              program_invocation_short_name, mpierrstr);
      MPI_Abort(mpicomm, EXIT_FAILURE);
    }
    int threadlevel;
    mpierrcode = MPI_Query_thread(&threadlevel);
    if (mpierrcode)
    {
      MPI_Error_string(mpierrcode, mpierrstr, &mpierrstrlen);
      fprintf(stderr, "%s: MPI_Query_thread failed with %s\n",
              program_invocation_short_name, mpierrstr);
      MPI_Abort(mpicomm, EXIT_FAILURE);
    }
    fprintf(stderr, "MPI version %s (thread level: ", mpiversionstr);
    if (threadlevel == MPI_THREAD_SINGLE)
      fprintf(stderr, "MPI_THREAD_SINGLE");
    else if (threadlevel == MPI_THREAD_FUNNELED)
      fprintf(stderr, "MPI_THREAD_FUNNELED");
    else if (threadlevel == MPI_THREAD_SERIALIZED)
      fprintf(stderr, "MPI_THREAD_SERIALIZED");
    else if (threadlevel == MPI_THREAD_MULTIPLE)
      fprintf(stderr, "MPI_THREAD_MULTIPLE");
    else
      fprintf(stderr, "unknown");
    fprintf(stderr, ")\n");
  }

  /* 1b) parse program options */
  struct program_options args;
  err = program_options_init(&args);
  errexit = err;
  MPI_Allreduce(MPI_IN_PLACE, &errexit, 1, MPI_C_BOOL, MPI_LOR, mpicomm);
  if (errexit)
  {
    if (err)
      fprintf(stderr, "%s:%d: %s\n", program_invocation_short_name, __LINE__, strerror(err));
    MPI_Abort(mpicomm, EXIT_FAILURE);
  }

  int nargs;
  err = parse_program_options(argc, argv, &args, &nargs);
  errexit = err;
  MPI_Allreduce(MPI_IN_PLACE, &errexit, 1, MPI_C_BOOL, MPI_LOR, mpicomm);
  if (err)
  {
    if (rank == root)
    {
      fprintf(stderr, "%s: %s %s\n", program_invocation_short_name,
              strerror(err), argv[nargs]);
    }
    program_options_free(&args);
    MPI_Abort(mpicomm, EXIT_FAILURE);
  }
  MPI_Allreduce(MPI_IN_PLACE, &args.help, 1, MPI_C_BOOL, MPI_LOR, mpicomm);
  if (args.help)
  {
    if (rank == root)
      program_options_print_help(stdout);
    program_options_free(&args);
    MPI_Finalize();
    return EXIT_SUCCESS;
  }
  MPI_Allreduce(MPI_IN_PLACE, &args.version, 1, MPI_C_BOOL, MPI_LOR, mpicomm);
  if (args.version)
  {
    if (rank == root)
      program_options_print_version(stdout);
    program_options_free(&args);
    MPI_Finalize();
    return EXIT_SUCCESS;
  }
  errexit = !args.Apath;
  MPI_Allreduce(MPI_IN_PLACE, &errexit, 1, MPI_C_BOOL, MPI_LOR, mpicomm);
  if (errexit)
  {
    if (rank == root)
      program_options_print_usage(stdout);
    program_options_free(&args);
    MPI_Finalize();
    return EXIT_FAILURE;
  }
  errexit = args.bpath && args.manufactured_solution;
  MPI_Allreduce(MPI_IN_PLACE, &errexit, 1, MPI_C_BOOL, MPI_LOR, mpicomm);
  if (errexit)
  {
    if (rank == root)
      program_options_print_usage(stdout);
    program_options_free(&args);
    MPI_Finalize();
    return EXIT_FAILURE;
  }

  const char *Apath = args.Apath;
  const char *bpath = args.bpath;
  int quiet = args.quiet;
  int verbose = args.verbose;
  const char *numfmt = args.numfmt;
  acgidx_t seed = args.seed;
  double diffatol = args.diffatol;
  double diffrtol = args.diffrtol;
  double residualatol = args.residualatol;
  double residualrtol = args.residualrtol;
  int maxits = args.maxits;
  int output_comm_matrix = args.output_comm_matrix;

  /* read matrix A */
  struct acgsymcsrmatrix A;
  // ... (Matrix reading logic would go here, simplified for skeleton)
  // For now, let's assume we have A, b, x initialized or we just skip to solver
  // init for skeleton

  /* initialize SYCL queue */
  sycl::queue q;
  try
  {
    q = sycl::queue(sycl::default_selector_v);
  }
  catch (sycl::exception const &e)
  {
    std::cerr << "SYCL exception caught: " << e.what() << std::endl;
    return 1;
  }

  if (!args.quiet && rank == 0)
  {
    std::cout << "Running on "
              << q.get_device().get_info<sycl::info::device::name>()
              << std::endl;
  }

  /* solve */
  if (args.solvertype == acgsolver_acg)
  {
    struct acgsolversycl cg;
    // err = acgsolversycl_init(&cg, &A, q, &comm);
    // if (err) { ... }
    // err = acgsolversycl_solve(&cg, &A, &b, &x, ...);
  }

  program_options_free(&args);

#ifdef ACG_HAVE_PETSC
  PetscFinalize();
#endif
#ifdef ACG_HAVE_MPI
  MPI_Finalize();
#endif

  return EXIT_SUCCESS;
}
