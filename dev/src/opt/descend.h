#pragma once

#include "itensor/all_basic.h"
#include "itensor/util/print_macro.h"

#include "../util/database.h"
#include "../util/extract_mat.h"
#include "../util/cft_data.h"
#include "../util/glue.h"

using namespace itensor;
using namespace std;

// matrix log of a square tensor
// the up and down indexsets have tags "u" and "d"
ITensor tensor_log(ITensor A);

void local_gauge(int n_chain);