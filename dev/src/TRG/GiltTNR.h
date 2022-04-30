#pragma once

#include "../inc.h"

using namespace itensor;
using namespace std;

class GiltTNR
{
public:
	ITensor A;
	ITensor A1, A2;
	Index A1u, A1r, A1d, A1l, A2u, A2r, A2d, A2l;
	//ITensor Rp;
	IndexSet edge_is; // record the cut indexSet of environment of current edge
	IndexSet is_svd; // record indexSet of optimize_bare


	double gilt_eps;// The threshold for how small singular values are considered "small enough" in Gilt
	double convergence_eps = 1E-2;
	double log_fact = 0;

	int chi_trg;
	double trg_cutoff;

	double gilt_error = 0;
	std::map<const char, bool> flag;

public:
	GiltTNR(ITensor& A_);

	void gilttnr_step();

	void gilt_plaq();

	void rotate_edge(const char c, bool reverse = false);

	// gilt truncate a leg
	void apply_gilt(const char);

	std::tuple<ITensor, ITensor> get_envspec(const char);

	ITensor optimize_Rp(ITensor U, ITensor D);
	std::tuple<ITensor, ITensor, ITensor> optimize_Rp_bare(ITensor U, ITensor D);
};

