#include "glue.h"


// glue a row of 4-leg tensors A with tags "u","r","d","l"
// output a chain MPO
ITensor glue_bare(const ITensor& A, int n) {

	ITensor B = A;

	if (n == 1) return B;

	else if (n > 1) {
		for (int i = 1; i < n; i++) {
			// contract with A
			ITensor Ai = prime(A, i);
			B *= delta(findIndex(B, "r"), findIndex(Ai, "l"));
			B *= Ai;
		}

		return B;
	}
	
}

// glue a row of 4-leg tensors A with tags "u","r","d","l"
// output a ring MPO
ITensor glue_bare_ring(const ITensor& A, int n) {
	ITensor B = glue_bare(A, n);
	B *= delta(findIndex(B, "l"), findIndex(B, "r"));
	return B;
}


// glue a ring of 4-leg tensors A with tags "u","r","d","l"
// output a rank-2 tensor
ITensor glue(const ITensor& A, int n, bool twist) {
	if (n == 1) return A;

	int nl = 1;
	int nr = 0;
	if (n % 2 == 0) nl = n / 2;
	else nl = (n - 1) / 2;
	nr = n - nl;

	ITensor AL = glue_bare(A, nl);
	ITensor AR = glue_bare(A, nr);

	// contract AL and AR
	AR.prime(nl);

	Index ALl = findIndex(AL, "l");
	Index ALr = findIndex(AL, "r");
	Index ARl = findIndex(AR, "l");
	Index ARr = findIndex(AR, "r");

	AL.replaceInds({ ALl,ALr }, { ARr,ARl });
	ITensor res = AL * AR;

	IndexSet u_is = findInds(res, "u");
	IndexSet d_is = findInds(res, "d");
	auto [uT, U] = combiner(u_is);
	auto [dT, D] = combiner(d_is);

	// swap the first "up" index with the rest "up" indices
	if (twist) {
		Index u = findIndex(A, "u"); //this unprimed index is also shared by res

		res.addTags("first", { u });
		IndexSet u_rest = findIndsExcept(findInds(res, "u"), "first");
		res.removeTags("first");

		res.swapInds({u}, u_rest);
	}
	
	res *= uT;
	res *= dT;

	return res;
}