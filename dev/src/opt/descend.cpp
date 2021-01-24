#include "descend.h"

// matrix log of a square tensor
void tensor_log() {
	const double beta_c = 0.5 * log(1 + sqrt(2));
	ITensor A0 = database::ising2d(beta_c);
	//A0.randomize();
	constexpr int n_chain = 2;

	ITensor A = glue_bare_ring(A0, n_chain);
	//PrintData(A);

	IndexSet u_is = findInds(A, "u");
	IndexSet d_is = findInds(A, "d");
	auto [uT, U] = combiner(u_is);
	auto [dT, D] = combiner(d_is);

	A = uT * A * dT;//now A has order 2
	PrintData(A);
	int di = A.index(1).dim();
	int dj = A.index(2).dim();

	auto extractReal = [](Dense<Real> const& d)
	{
		return d.store;
	};

	auto data_vec = applyFunc(extractReal, A.store());

	arma::mat denseT(&data_vec[0], di, dj, true);
	denseT.print("denseT");

	denseT = real(arma::logmat(denseT));
	denseT.print("denseT");

	PrintData(A);


	PrintData(A);

	//todo: obtain an ITensor from arma mat

	//todo: how to directly change the stored memory of an ITensor

	//todo: match indices, then contract with uT and dT to restore index structure
}