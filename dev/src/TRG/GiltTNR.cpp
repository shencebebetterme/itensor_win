#include "pch.h"
#include "GiltTNR.h"
#include "trg.h"
#include "../util/it_help.h"


GiltTNR::GiltTNR(ITensor& A_)
{
	A = A_;
	chi_trg = 9;
	gilt_eps = 4E-9;
	trg_cutoff = 1E-10;

	flag = {
		{'N',false},
		{'E',false},
		{'S',false},
		{'W',false}
	};
}

void GiltTNR::gilttnr_step() {
	A /= norm(A);

	A1 = A;
	A2 = A;

	A1.addTags("A1");
	A2.addTags("A2");
	A1u = findIndex(A1, "u");
	A1r = findIndex(A1, "r");
	A1d = findIndex(A1, "d");
	A1l = findIndex(A1, "l");

	A2u = findIndex(A2, "u");
	A2r = findIndex(A2, "r");
	A2d = findIndex(A2, "d");
	A2l = findIndex(A2, "l");

	if (gilt_eps > 0) {
		this->gilt_plaq();
	}

	printf("\n beginning trg steps\n");
	//PrintData(A);
	A = trg_step(A1, A2, chi_trg, trg_cutoff);
	//PrintData(A);
	A = trg_step(A, A, chi_trg, trg_cutoff);
	//PrintData(A);
	printf(" trg steps finished\n");
}


// all 4 legs around the plaquette are truncated
// A1 and A2 are updated in this step
void GiltTNR::gilt_plaq() {

	gilt_error = 0;

	flag = {
	{'N',false},
	{'E',false},
	{'S',false},
	{'W',false}
	};

	printf("\noriginal shape is ");
	show_shape(A1);
	printf("\n");

	while (1) {
		this->apply_gilt('S');
		this->apply_gilt('N');
		this->apply_gilt('E');
		this->apply_gilt('W');

		if (flag['N'] && flag['E'] && flag['S'] && flag['W']) break;
	}
}


//todo: speed up optimize_Rp
// apply gilt to a single leg labeled by char
// char can be one of { 'N', 'E', 'S', 'W' }
void GiltTNR::apply_gilt(const char c) {
	printf("applying gilt %c, ", c);

	auto [U, D] = this->get_envspec(c);
	//Print(U);

	ITensor Rp;
	Rp = this->optimize_Rp(U, D);
	//Print(Rp);

	// split Rp by factor()
	double split_eps = gilt_eps * 1E-3;
	
	// the indices of Rp is recorded in edge_is
	// always attach Rpl to A2
	auto [Rp_l, Rp_r] = factor(Rp, { findIndex(Rp,"A2") }, { "Tags=","rp","SVDMethod=","gesdd","Cutoff=",split_eps });
	auto [u_, s_, v_] = svd(Rp, { findIndex(Rp,"A2") }, { "SVDMethod=","gesdd","Cutoff=",split_eps });

	ITensor d2id = apply(s_, [](double z) {return z - 1; });//distance to identity
	if (get_max_abs(d2id) < convergence_eps) {
		flag[c] = true;
	}

	//todo: gilt_error += insertionerr + spliterr
	// 
	// absorb into A1 and A2
	if (c == 'N') {
		A2 *= Rp_l;
		A2.replaceTags("rp", "A2,r");
		A2r = findIndex(A2, "r");
		A1 *= Rp_r;
		A1.replaceTags("rp", "A1,l");
		A1l = findIndex(A1, "l");
	}
	else if (c == 'E') {
		A2 *= Rp_l;
		A2.replaceTags("rp", "A2,u");
		A2u = findIndex(A2, "u");
		A1 *= Rp_r;
		A1.replaceTags("rp", "A1,d");
		A1d = findIndex(A1, "d");
	}
	else if (c == 'S') {
		A2 *= Rp_l;
		A2.replaceTags("rp", "A2,l");
		A2l = findIndex(A2, "l");
		A1 *= Rp_r;
		A1.replaceTags("rp", "A1,r");
		A1r = findIndex(A1, "r");
	}
	else if (c == 'W') {
		A2 *= Rp_l;
		A2.replaceTags("rp", "A2,d");
		A2d = findIndex(A2, "d");
		A1 *= Rp_r;
		A1.replaceTags("rp", "A1,u");
		A1u = findIndex(A1, "u");
	}
	//Print(A1); //Print(A2);
	printf("  current shape is ");
	show_shape(A1);
	printf("\n");
}


// rotate or anti-rotate
void GiltTNR::rotate_edge(const char c, bool reverse) {
	if (c == 'N'); // N by default

	if (c == 'E') {
		if (!reverse) {
			A2l = findIndex(A1, "u");
			A2u = findIndex(A1, "r");
			A2r = findIndex(A1, "d");
			A2d = findIndex(A1, "l");
			A1l = findIndex(A2, "u");
			A1u = findIndex(A2, "r");
			A1r = findIndex(A2, "d");
			A1d = findIndex(A2, "l");
			swap(A1, A2);
		}
		else {
			swap(A1, A2);
			A2l = findIndex(A2, "l");
			A2u = findIndex(A2, "u");
			A2r = findIndex(A2, "r");
			A2d = findIndex(A2, "d");
			A1l = findIndex(A1, "l");
			A1u = findIndex(A1, "u");
			A1r = findIndex(A1, "r");
			A1d = findIndex(A1, "d");
		}
	}

	if (c == 'S') {
		swap(A1u, A2u);
		swap(A1r, A2r);
		swap(A1d, A2d);
		swap(A1l, A2l);
		swap(A1, A2);
	}

	if (c == 'W') {
		if (!reverse) {
			A2r = findIndex(A1, "u");
			A2d = findIndex(A1, "r");
			A2l = findIndex(A1, "d");
			A2u = findIndex(A1, "l");
			A1l = findIndex(A2, "d");
			A1u = findIndex(A2, "l");
			A1r = findIndex(A2, "u");
			A1d = findIndex(A2, "r");
			swap(A1, A2);
		}
		else {
			swap(A1, A2);
			A2l = findIndex(A2, "l");
			A2u = findIndex(A2, "u");
			A2r = findIndex(A2, "r");
			A2d = findIndex(A2, "d");
			A1l = findIndex(A1, "l");
			A1u = findIndex(A1, "u");
			A1r = findIndex(A1, "r");
			A1d = findIndex(A1, "d");
		}
	}
}



// A1 and A2 indices should have tags "A1" and "A2" respectively
// use eig of doubled environment rather than the costly svd
// return the singular vectors and singular values
std::tuple<ITensor, ITensor> GiltTNR::get_envspec(const char c) {
	ITensor env;
	ITensor env_db; //doubled env
	
	this->rotate_edge(c, false);

	edge_is = { A2r,A1l };
	//Print(is);

	ITensor env_db_down_l = replaceInds(A1, { A1l,A1d }, { prime(A1l),prime(A1d) }) * prime(conj(A1));
	ITensor env_db_down_r = replaceInds(A2, { A2r,A2d }, { prime(A2r),prime(A2d) }) * prime(conj(A2));
	ITensor env_db_up_l = replaceInds(A2, { A2l,A2u }, { prime(A2l),prime(A2u) }) * prime(conj(A2));
	ITensor env_db_up_r = replaceInds(A1, { A1r,A1u }, { prime(A1r),prime(A1u) }) * prime(conj(A1));
	env = env_db_up_l * replaceInds(env_db_down_l, { A1u, prime(A1u), A1r,prime(A1r) }, { A2d,prime(A2d), A2l,prime(A2l) }) * replaceInds(env_db_down_r, { A2u,prime(A2u) }, { A1d,prime(A1d) }) * env_db_up_r;
	//Print(env);

	this->rotate_edge(c, true);
	//Print(A1); Print(IndexSet({ A1u,A1r,A1d,A1l }));
	//Print(A2); Print(IndexSet({ A2u,A2r,A2d,A2l }));

	//Print(env);
	auto[U,D] = diagHermitian(env);
	
	// eigenvalue of doubled env --> singular value
	auto lambda = [](double z) {return sqrt(abs(z)); };
	D.apply(lambda);

	return { U,D };
}



// return the factored ouput Rp_left, Rp_right and the singular values
std::tuple<ITensor, ITensor, ITensor> GiltTNR::optimize_Rp_bare(ITensor U, ITensor D) {
	D /= get_sum(D);

	ITensor t = U * delta(U.index(1), U.index(2));
	// weight has the same index structure as D
	auto get_weight = [=](double z) {return pow(z, 2) / (pow(z, 2) + pow(gilt_eps, 2)); };
	ITensor tp = t * apply(D, get_weight);
	//Print(t); Print(tp);
	//tp.setPrime(primeLevel(t.index(1)));
	tp.replaceInds({ tp.index(1) }, { t.index(1) }); // for contraction with U
	ITensor Rp = tp * conj(U);
	//Print(Rp);

	double split_eps = gilt_eps * 1E-3;
	auto [u_, s_, v_] = svd(Rp, { Rp.index(1) }, { "SVDMethod=","gesdd","Cutoff=",split_eps });
	auto [Rp_l, Rp_r] = factor(Rp, { Rp.index(1) }, { "SVDMethod=","gesdd","Cutoff=",split_eps,"Tags=","Rp,l" });
	//Print(Rp_l); Print(Rp_r);
	Index id_Rp_c = commonIndex(Rp_l, Rp_r);
	Rp_r.replaceTags("l", "r", "Rp");
	is_svd = IndexSet({ id_Rp_c,replaceTags(id_Rp_c,"l","r") });

	return { Rp_l,Rp_r,s_ };
}


//todo: accelerate this part
// given the environment singular vector U and singular values D
// choose t' and build R'
// Rp should have the same indices as the original R
ITensor GiltTNR::optimize_Rp(ITensor U, ITensor D) {

	std::vector<ITensor> Rp_l_vec = {};
	std::vector<ITensor> Rp_r_vec = {};
	ITensor U_i = U;
	ITensor D_i = D;
	ITensor tmp;
	bool done_recursing = false;
	double diff = 0;

	auto Tm1 = [](double z) {return z - 1; };
	

	while (!done_recursing) {
		//Print(U_i);
		auto [Rp_l, Rp_r, s_] = optimize_Rp_bare(U_i, D_i);
		Rp_l_vec.push_back(Rp_l);
		Rp_r_vec.push_back(Rp_r);

		//ITensor d2id = apply(s_, [](double z) {return z - 1; });//distance to identity
		// diff grows near 1 and then drops, 
		// the larger singular vallues are all near 1
		// the smallest singular value decreases to 0 and gets abandoned in the next optimization : bond truncation
		diff = get_max_abs(s_.apply(Tm1)) - convergence_eps;
		done_recursing = diff < 0;
		// Rp_l and Rp_r will not contract 
		ITensor UusvS = (Rp_l * U_i * Rp_r) * D_i;

		//Print(Rp_l); Print(Rp_r);
		std::tie(U_i, D_i, tmp) = svd(UusvS, is_svd, { "SVDMethod=","gesdd" });
	}

	// reconstruct Rp_L and Rp_R from vector
	ITensor Rp_L = trivialIT();
	for (auto rl : Rp_l_vec) {
		Rp_L *= rl;
	}
	Rp_L.removeTags("l","Rp");

	ITensor Rp_R = trivialIT();
	for (auto rl : Rp_r_vec) {
		Rp_R *= rl;
	}
	Rp_R.removeTags("r","Rp");
	//Print(Rp_R);
	//remove the tag "l" and "r" in Rp_l and Rp_r at the end point, s.t. they can contract later

	return Rp_L * Rp_R;

	//iterative for loop
#if 0
	//iteration for loop
	double split_eps = gilt_eps * 1E-3;
	auto [u_, s_, v_] = svd(Rp, { Rp.index(1) }, { "SVDMethod=","gesdd","Cutoff=",split_eps });
	auto [Rp_l, Rp_r] = factor(Rp, { Rp.index(1) }, { "SVDMethod=","gesdd","Cutoff=",split_eps });


	IndexSet is_svd;

	ITensor UusvS = Rp_l * U * D * Rp_r;
	auto [U_inner, D_inner, V_inner] = svd(UusvS, is_svd, { "SVDMethod=","gesdd" });

	ITensor Rpinner = optimize_Rp(U_inner, D_inner);
	//todo: tweak indices
	Rp = Rpinner * Rp_l * Rp_r;
#endif

}