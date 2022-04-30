#include "pch.h"
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
	arma::cx_mat denseT = extract_cxmat(A);
	//stupid workaround of degenerate eigenvalues
	denseT += 1E-15 * randu<cx_mat>(denseT.n_rows, denseT.n_cols);
	denseT = arma::logmat(denseT);
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


void local_gauge(int n_chain) {
	const double beta_c = 0.5 * log(1 + sqrt(2));
	ITensor A0 = database::ising2d(beta_c);
	const double c_ising = 0.5;//central charge
	const double f_ising = 0.929695;//free energy per site
	const double fA = f_ising * 2; //free energy per A tensor

	//constexpr int n_chain = 4;
	ITensor MT = glue(A0, n_chain, true, false);
	//Print(MT);
	double factor = std::exp((2 * PI / n_chain) * (c_ising / 12) + n_chain * fA);
	MT /= factor; // remove central charge and free energy
	ITensor logMT = tensor_log(MT);

	//obtain L0+L0bar and L0-L0bar
	ITensor sum = -(n_chain / (2 * PI)) * realPart(logMT);
	ITensor diff = (n_chain / (2 * PI)) * imagPart(logMT);
	//PrintData(sum); PrintData(diff);
	ITensor L0 = (sum - diff) / 2;
	Print(L0);

	ITensor Lm1 = L0; //L-1
	//apply rotation matrix to down legs according to prime level
	IndexSet Uis = findInds(L0, "u");
	IndexSet Dis = findInds(L0, "d");
	for (auto id : Dis) {
		double theta = 2 * PI * primeLevel(id) / n_chain;
		double cos_theta = std::cos(theta);
		double sin_theta = std::sin(theta);

		Index idp = addTags(id, "ex");
		ITensor gi(id, idp);
		// apply gauge
		if (dim(id) == 2) {
			gi.set(id = 1, idp = 1, cos_theta);
			gi.set(id = 1, idp = 2, -sin_theta);
			gi.set(id = 2, idp = 1, sin_theta);
			gi.set(id = 2, idp = 2, cos_theta);
		}
		Lm1 *= gi;
	}
	Lm1.removeTags("ex");

	//change L0 and L1 tensor into matrix
	auto [uT, Uidx] = combiner(Uis);
	auto [dT, Didx] = combiner(Dis);
	L0 = uT * L0 * dT;
	Lm1 = uT * Lm1 * dT;
	auto L0mat = extract_cxmat(L0);
	auto Lm1mat = extract_cxmat(Lm1);
	//L0mat.print("L0mat"); Lm1mat.print("L1mat");

	arma::cx_mat comm = Lm1mat * L0mat - L0mat * Lm1mat;
	double reldiff = norm(comm - Lm1mat)/norm(comm);
	std::cout << "\nrelative diff = " << reldiff << std::endl;
}