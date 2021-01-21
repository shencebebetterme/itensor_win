#include "descend.h"

// matrix log of a square tensor
void tensor_log() {
	const double beta_c = 0.5 * log(1 + sqrt(2));
	ITensor A0 = database::ising2d(beta_c);

	ITensor A = glue_bare_ring(A0, 3);
}