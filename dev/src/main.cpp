
#include "pch.h"
#include "util/database.h"
#include "util/extract_mat.h"
#include "util/cft_data.h"
#include "util/glue.h"
#include "util/it_help.h"

#include "TRG/tnr.h"
#include "TRG/trg.h"
#include "TRG/GiltTNR.h"

#include "opt/descend.h"


//ITensor trg(ITensor, int, int);

// #include "sample/ctmrg.h"
// #include "itensor/all.h"
// #include "util/arnoldi.h"

// TODO todo


int main() {
	Index i(2, "i");
	Index j(2, "j");          
	IndexSet indices({ i, j }); 
	ITensor T(indices);
	T.set(i(1), j(1), 1.0);
	T.set(i(2), j(2), 1.0);

	auto vec_ni = (*((ITWrap<Dense<double>>*) & (*T.store()))).d.store;

}