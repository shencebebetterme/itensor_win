#pragma once

#include "pch.h"

#include "inc.h"
#include "util/database.h"
#include "util/extract_mat.h"
#include "util/cft_data.h"
#include "util/glue.h"
#include "util/it_help.h"

#include "util/arpack_wrap.h"

class adITensorMap : public ITensorMapBase {
public:

	ITensor Z_; // to implement ad_Z
	int ldim_; // dim of each leg
	int nl_; // Z has 2*nl legs
	IndexSet uis;
	IndexSet dis;
	IndexSet tis; //temporary index sets

	// inds(Z_) = inds(x) = is = {u, u', u'', ... d, d', d'', ...}
	adITensorMap(IndexSet& is)
		: ITensorMapBase(is)
	{}


	// ad_Z maps X to [X,Z]
	void ad(ITensor const& x, ITensor& y) const 
	{
		// y = Z x - x Z
		// suppose x and Z has the same indexing structure
		// with indices u, u', u'', ... and d, d', d'', ...
		ITensor Zc = Z_;
		ITensor xc = x;
		Zc.replaceTags("u", "uu").replaceTags("d", "u");
		xc.replaceTags("u", "uu").replaceTags("d", "u");
		y = Zc * x - xc * Z_; 
		y.replaceTags("uu", "u"); // now y has indices u and d
	}

	// ad^2
	void product(ITensor const& x, ITensor& y) const
	{
		ITensor tmp = x;
		ad(x, tmp);
		ad(tmp, y);
		//y *= -1;
	}
};



void adtest1() {
	int dim0 = 2;
	Index d(dim0, "d");
	Index u = replaceTags(d, "d", "u");
	//ITensor sz(u,d);
	//sz.set(1, 1, 1.0);
	//sz.set(2, 2, -1.0);
	
	
	//PrintData(sz);

	IndexSet is = { u,prime(u),d,prime(d) };
	adITensorMap amap(is);

	ITensor ssz(is);
	ssz.set(1, 1, 1, 1, 1.0);
	ssz.set(2, 2, 2, 2, -1.0);

	amap.Z_ = ssz;

	int nev = 3;
	std::vector<Cplx> eigval = {};
	std::vector<ITensor> eigvecs = {};
	itwrap::eig_arpack<double>(eigval, eigvecs, amap, { "nev=",nev,"tol=",1E-8, "ReEigvec=",true,"WhichEig=","SM" });
}



void adtest2() {
	const double beta_c = 0.5 * log(1 + sqrt(2));
	ITensor A0 = database::ising2d(beta_c);

	Index d = findIndex(A0, "d");
	Index u = findIndex(A0, "u");

	int nl_ = 4;
	std::vector<Index> idv = {};
	for (int i = 0; i < nl_; i++) {
		Index ui = prime(u, i);
		idv.push_back(ui);
	}
	for (int i = 0; i < nl_; i++) {
		Index di = prime(d, i);
		idv.push_back(di);
	}

	IndexSet is = idv;
	adITensorMap amap(is);
}