#include "extract_mat.h"

/*
arma::sp_mat extract_spmat(const ITensor& T) {
    auto di = T.index(1).dim();
    auto dj = T.index(2).dim();

    auto extractReal = [](Dense<Real> const& d)
    {
        return d.store;
    };

    auto data_vec = applyFunc(extractReal, T.store());

    arma::mat denseT(&data_vec[0], di, dj, false);
    arma::sp_mat sparseT(denseT);
    return sparseT;
}
*/


//// directly copy the chunk of memory, index order verified
//// copy twice
//arma::mat extract_mat(const ITensor& T) {
//    auto di = T.index(1).dim();
//    auto dj = T.index(2).dim();
//
//    auto extractReal = [](Dense<Real> const& d)
//    {
//        return d.store;
//    };
//
//    //the data is already copied to data_vec
//    //copying data to this vector is the most time-consuming step
//    auto data_vec = applyFunc(extractReal, T.store());
//
//    arma::mat denseT(&data_vec[0], di, dj, false);
//    return denseT;
//}


// as fast as the following copy=true
// but the data can be modified
//arma::mat extract_mat(const ITensor& T) {
//	auto di = T.index(1).dim();
//	auto dj = T.index(2).dim();
//
//	auto pt = &((*((ITWrap<Dense<double>>*) & (*T.store()))).d.store[0]);
//    arma::mat denseT(pt, di, dj, false);
//    return denseT;
//}


// if copy=true, then there's one copy
// if copy=false, then there's zero copy
arma::mat extract_mat(ITensor& T, bool copy) {
    
    if (!isReal(T)) {
        Error("Tensor not real!");
    }

	auto di = T.index(1).dim();
	auto dj = T.index(2).dim();

	auto pt = &((*((ITWrap<Dense<double>>*) & (*T.store()))).d.store[0]);
	//arma::mat* denseT = new arma::mat(pt, di, dj, copy);
    // the final data in denseT is moved, so the overhead is negligible
    arma::mat denseT(pt, di, dj, copy);

	return denseT;
}



arma::cx_mat extract_cxmat(ITensor& T, bool copy) {

	if (!isComplex(T)) {
        Error("Tensor not complex!");
	}

	auto di = T.index(1).dim();
	auto dj = T.index(2).dim();

	auto pt = &((*((ITWrap<Dense<Cplx>>*) & (*T.store()))).d.store[0]);
	//arma::cx_mat* denseT = new arma::cx_mat(pt, di, dj, copy);
    arma::cx_mat denseT(pt, di, dj, copy);

	return denseT;
}




// very difficult to extract itensor with similar method
// need to overload the allocator of vec_no_init

// extract a rank-2 ITensor from an arma::mat
ITensor extract_it(arma::mat& M) {
    int nr = M.n_rows;
    int nc = M.n_cols;
    Index i(nr, "i");
    Index j(nc, "j");
    ITensor A(i, j);

    for(auto ir:range(nr))
        for (auto ic : range(nc)) {
            double val = M(ir, ic);
            A.set(i = ir+1, j = ic+1, val);
        }

    return A;
}


// extract a rank-2 ITensor from an arma::cx_mat
ITensor extract_it(arma::cx_mat& M) {
	int nr = M.n_rows;
	int nc = M.n_cols;
	Index i(nr, "i");
	Index j(nc, "j");
	ITensor A(i, j);

	for (auto ir : range(nr))
		for (auto ic : range(nc)) {
			Cplx val = M(ir, ic);
			A.set(i = ir + 1, j = ic + 1, val);
		}

	return A;
}