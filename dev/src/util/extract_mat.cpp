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


arma::mat extract_mat(const ITensor& T) {
    auto di = T.index(1).dim();
    auto dj = T.index(2).dim();

    auto extractReal = [](Dense<Real> const& d)
    {
        return d.store;
    };

    auto data_vec = applyFunc(extractReal, T.store());

    arma::mat denseT(&data_vec[0], di, dj, true);
    return denseT;
}


arma::cx_mat extract_cxmat(const ITensor& T) {
    Index i = T.index(1);
    Index j = T.index(2);
	auto di = i.dim();
	auto dj = j.dim();

    arma::cx_mat denseT(di, dj);

	for (auto ir : range(di))
		for (auto ic : range(dj)) {
            Cplx val = eltC(T, i = ir + 1, j = ic + 1);
            denseT(ir, ic) = val;
		}

    return denseT;
}


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