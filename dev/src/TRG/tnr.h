#pragma once

#include "itensor/all_basic.h"
#include "itensor/util/print_macro.h"
using namespace itensor;
using namespace std;

//ITensor update(const ITensor& AA, const IndexSet& is, const int maxdim);

class TNR
{
public:
	ITensor A, A4;
	ITensor U, Uc, vL, vLc, vR, vRc;
	ITensor B, Bc, C, B_lower, BBl, BBlU;
	ITensor Gamma_U, Gamma_vL, Gamma_vR;
	ITensor z, zc;
	Index u, r, d, l;
	Index ul, ur, dl, dr;
	Index Li, Lo1, Lo2, Ri, Ro1, Ro2;

	ITensor tmp, tmp1, tmp2;

	int max_iter = 300;
	//double err_bound = 5E-3;
	double norm_change_bound = 1E-9;
	double B_norm;
	bool horz_refl = true;

public:
	int chi_tnr = 1;
	int chi_trg = 1;
	int chi_trg_B = 1;
	int chi_trg_C = 1;

	TNR(ITensor& A_);



	void buildA4();
	void calculateB();
	void calculateC();
	void calculateB_lower();

	void build_uvw();
	void initial_uvw();
	void optimize_isometries();
	void optimize_disentangler();

	void optimize_uvw();
	void build_z_and_A();


	void trg_step();
	void tnr_step();

};