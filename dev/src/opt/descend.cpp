#include "descend.h"

// matrix log of a square tensor
// the up and down indexsets have tags "u" and "d"
ITensor tensor_log(ITensor A) {

	IndexSet u_is = findInds(A, "u");
	IndexSet d_is = findInds(A, "d");
	auto [uT, U] = combiner(u_is);
	auto [dT, D] = combiner(d_is);

	A = uT * A * dT;//now A has order 2
	//PrintData(A);
	int di = A.index(1).dim();
	int dj = A.index(2).dim();

	auto extractReal = [](Dense<Real> const& d)
	{
		return d.store;
	};

	auto data_vec = applyFunc(extractReal, A.store());

	arma::mat denseT(&data_vec[0], di, dj, true);
	//denseT.print("denseT");

	denseT = real(arma::logmat(denseT));
	//denseT.print("denseT");


	ITensor logA = extract_it(denseT);
	//PrintData(logA);
	//todo: how to directly change the stored memory of an ITensor

	//todo: match indices, then contract with uT and dT to restore index structure
	//restore indices
	logA.replaceInds({ logA.index(1),logA.index(2) }, { A.index(1),A.index(2) });
	logA = uT * logA * dT;
	//PrintData(logA);

	return logA;
}