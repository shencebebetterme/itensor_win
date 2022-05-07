#pragma once


//#define ARMA_BLAS_NO_UNDERSCORE
#define ARMA_DONT_USE_FORTRAN_HIDDEN_ARGS
#define ARMA_USE_HDF5
#include <armadillo>

#include "itensor/all.h"
#include "itensor/util/print_macro.h"

#include <iostream>
#include <chrono>

using namespace itensor;
using namespace std;
using namespace std::chrono;
using namespace arma;

