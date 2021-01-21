#pragma once

#include "../inc.h"


// glue a row of 4-leg tensors A with tags "u","r","d","l"
// output a chain MPO
ITensor glue_bare(const ITensor& A, int n);

// glue a row of 4-leg tensors A with tags "u","r","d","l"
// output a ring MPO
ITensor glue_bare_ring(const ITensor& A, int n);

// glue a ring of 4-leg tensors A with tags "u","r","d","l"
// output a rank-2 tensor
ITensor glue(const ITensor& A, int n, bool twist = false);