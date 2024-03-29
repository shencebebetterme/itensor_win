#include "pch.h"
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
// need to overload the allocator of vec_no_init?
// and also the .memptr() of arma::mat is const
// so just copy the bunch of memory

// https://stackoverflow.com/questions/2434196/how-to-initialize-stdvector-from-c-style-array
// https://stackoverflow.com/questions/21917529/is-it-possible-to-initialize-stdvector-over-already-allocated-memory

// extract a rank-2 ITensor from an arma::mat
ITensor extract_it(const arma::mat& M) {
    int nr = M.n_rows;
    int nc = M.n_cols;

    Index i(nr, "i");
    Index j(nc, "j");
	ITensor A = setElt(0.0, i = 1, j = 1); // initialize a trivial ITensor

	vector_no_init<double>& dvec = (*((ITWrap<Dense<double>>*) & (*A.store()))).d.store;
	dvec.assign(M.begin(), M.end()); //copy the bunch of memory rather than set value element-by-element

    return A;
}


// extract a rank-2 ITensor from an arma::cx_mat
ITensor extract_it(const arma::cx_mat& M) {
	int nr = M.n_rows;
	int nc = M.n_cols;

	Index i(nr, "i");
	Index j(nc, "j");
	ITensor A = setElt(Cplx(0.0,1.0), i = 1, j = 1); // initialize a trivial ITensor

	vector_no_init<Cplx>& dvec = (*((ITWrap<Dense<Cplx>>*) & (*A.store()))).d.store;
	dvec.assign(M.begin(), M.end());

	return A;
}