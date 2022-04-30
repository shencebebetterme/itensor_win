#pragma once

#include "../pch.h"
#include "../util/database.h"
#include "../util/extract_mat.h"
#include "../util/cft_data.h"

ITensor trg(ITensor, int, int);

void ising_trg() {
	const double beta_c = 0.5 * log(1 + sqrt(2));
	ITensor A = db::ising2d(beta_c);
	ITensor As = trg(A, 20, 6); // maxdim=20, steps=6

	Index l = findIndex(As, "l");
	Index r = findIndex(As, "r");
	As *= delta(l, r);

	arma::mat TM = extract_mat(As);
	cd_dense(TM, 5, 1, 1); // show 5 states
}