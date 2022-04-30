#include "pch.h"
//#include "itensor/all_basic.h"
//using namespace itensor;

#include "database.h"

namespace database {


ITensor ising2d(double beta) {
	//const double beta_c = 0.5 * log(1 + sqrt(2));
	constexpr int dim0 = 2;
	Index s(dim0);

	Index u = addTags(s, "u");
	Index r = addTags(s, "r");
	Index d = addTags(s, "d");
	Index l = addTags(s, "l");

	ITensor A = ITensor(u, r, d, l);

	// Fill the A tensor with correct Boltzmann weights:
	// 1 -> 1
	// 2-> -1
	auto Sig = [](int s) { return 1. - 2. * (s - 1); };
	for (auto sl : range1(dim0))
		for (auto sd : range1(dim0))
			for (auto sr : range1(dim0))
				for (auto su : range1(dim0))
				{
					auto E = Sig(sl) * Sig(sd) + Sig(sd) * Sig(sr)
						+ Sig(sr) * Sig(su) + Sig(su) * Sig(sl);
					auto P = exp(E * beta);
					A.set(l(sl), r(sr), u(su), d(sd), P);
					//A.set(l(sl), r(sr), u(su), d(sd), 0);
				}
	return A;
}


// a Hadmard gate gauge multiplied to every leg of A
ITensor ising2d_gauge(double beta) {
	Index i(2, "i");
	Index j(2, "j");
	ITensor Hadmard(i, j);

	Hadmard.set(1, 1, 1.0);
	Hadmard.set(1, 2, 1.0);
	Hadmard.set(2, 1, 1.0);
	Hadmard.set(2, 2, -1.0);
	Hadmard /= std::sqrt(2);

	ITensor A = ising2d(beta);

	Index u = findIndex(A, "u");
	Index r = findIndex(A, "r");
	Index d = findIndex(A, "d");
	Index l = findIndex(A, "l");

	A *= delta(i, u) * Hadmard; A.replaceInds({ j }, { u });
	A *= delta(i, r) * Hadmard; A.replaceInds({ j }, { r });
	A *= delta(i, d) * Hadmard; A.replaceInds({ j }, { d });
	A *= delta(i, l) * Hadmard; A.replaceInds({ j }, { l });

	return A;
}


ITensor x4(const ITensor& A) {
	Index u = findIndex(A, "u");
	Index r = findIndex(A, "r");
	Index d = findIndex(A, "d");
	Index l = findIndex(A, "l");

	ITensor A2 = A * delta(r, prime(l)) * prime(A);
	ITensor A2p = prime(conj(A2), 2);

	ITensor A4 = A2 * delta(d, prime(u, 2));
	A4 *= delta(prime(d), prime(u, 3));
	A4 *= A2p;

	auto [U, iu] = combiner(findInds(A4, "u"), { "Tags=","u" });
	auto [R, ir] = combiner(findInds(A4, "r"), { "Tags=","r" });
	auto [D, id] = combiner(findInds(A4, "d"), { "Tags=","d" });
	auto [L, il] = combiner(findInds(A4, "l"), { "Tags=","l" });

	A4 *= U;
	A4 *= R;
	A4 *= D;
	A4 *= L;
	return A4;
}



ITensor fibo_avtx() {
	const double phi = (std::sqrt(5) + 1) / 2;

	constexpr int dim0 = 3;
	Index s(dim0);

	Index uc = addTags(s, "uc");//upper center
	Index ll = addTags(s, "ll");//lower left
	Index lr = addTags(s, "lr");//lower right

	ITensor aVtx(uc, lr, ll);

	for (int s_uc : range1(dim0))
		for (int s_lr : range1(dim0))
			for (int s_ll : range1(dim0))
			{
				double val = 0;
				//
				if (s_uc == 1 && s_lr == 1 && s_ll == 1) val = -std::pow(phi, -3.0 / 4);
				//
				if (s_uc == 2 && s_lr == 2 && s_ll == 1) val = std::pow(phi, 1.0 / 12);
				if (s_uc == 3 && s_lr == 1 && s_ll == 3) val = std::pow(phi, 1.0 / 12);
				if (s_uc == 1 && s_lr == 3 && s_ll == 2) val = std::pow(phi, 1.0 / 12);
				//
				aVtx.set(uc(s_uc), lr(s_lr), ll(s_ll), val);
			}
	return aVtx;
}




ITensor fibo_vtx() {
	const double phi = (std::sqrt(5) + 1) / 2;

	constexpr int dim0 = 3;
	Index s(dim0);

	Index lc = addTags(s, "lc");//lower center
	Index ul = addTags(s, "ul");//upper left
	Index ur = addTags(s, "ur");//upper right

	ITensor Vtx(ul, ur, lc);
	for (int s_ul : range1(dim0))
		for (int s_ur : range1(dim0))
			for (int s_lc : range1(dim0))
			{
				double val = 0;
				//
				if (s_ul == 1 && s_ur == 1 && s_lc == 1) val = -std::pow(phi, -3.0 / 4);
				//
				if (s_ul == 1 && s_ur == 2 && s_lc == 2) val = std::pow(phi, 1.0 / 12);
				if (s_ul == 3 && s_ur == 1 && s_lc == 3) val = std::pow(phi, 1.0 / 12);
				if (s_ul == 2 && s_ur == 3 && s_lc == 1) val = std::pow(phi, 1.0 / 12);
				//
				Vtx.set(ul(s_ul), ur(s_ur), lc(s_lc), val);
			}
	return Vtx;
}




ITensor fibo_vav() {
	ITensor vtx = fibo_vtx();
	Index lc = findIndex(vtx, "lc");

	ITensor avtx = fibo_avtx();
	Index uc = findIndex(avtx, "uc");

	ITensor vav = vtx * avtx * delta(lc, uc);
	return vav;
}




ITensor fibo_svd_bd() {
	ITensor vav = fibo_vav();
	Index ul = findIndex(vav, "ul");
	Index ur = findIndex(vav, "ur");
	Index ll = findIndex(vav, "ll");
	Index lr = findIndex(vav, "lr");

	auto [vL, vR] = factor(vav, { ul,ll }, { ur,lr }, { "MaxDim=",5,"Tags=","cr" });
	Index cr = commonIndex(vL, vR);
	Index cl = replaceTags(cr, "cr", "cl");
	vL *= delta(cr, cl); // vL has index cl, vR has index cr

	//ITensor result;
	ITensor vtx = fibo_vtx();
	ITensor avtx = fibo_avtx();

	//result = vR * delta(lr, findIndex(vtx, "ul")) * vtx;
	//result *= (vL * delta(ll, findIndex(vtx, "ur")));
	//result *= (delta(ur, findIndex(avtx, "ll")) * avtx * delta(ul, findIndex(avtx, "lr")));
	vR *= delta(lr, findIndex(vtx, "ul"));
	vR *= delta(ur, findIndex(avtx, "ll"));
	vL *= delta(ll, findIndex(vtx, "ur"));
	vL *= delta(ul, findIndex(avtx, "lr"));
	ITensor result = vR * vtx * vL * avtx;

	result.replaceTags("cr", "l");
	result.replaceTags("cl", "r");
	result.replaceTags("uc", "u");
	result.replaceTags("lc", "d");
	return result;
}

}