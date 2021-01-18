#pragma once

#include "itensor/all_basic.h"
#include "itensor/util/print_macro.h"
using namespace itensor;

ITensor trg(ITensor A, int maxdim, int scale);



//Apply the TRG algorithm to a checker-board lattice of tensors A1 and A2
// return the coarse grained A
ITensor trg_step(ITensor A1, ITensor A2, int maxdim, double cutoff);