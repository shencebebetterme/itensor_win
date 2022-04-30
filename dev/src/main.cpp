
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

//#include "sample/ctmrg.h"
//#include "itensor/all.h"
//#include "util/arnoldi.h"


int main() {
	arma::mat A = arma::randu(5, 5);
	print(A);
	//arma::cx_mat B = arma::logmat(A);
	ITensor T = extract_it(A);
	
	ITensor M = db::ising2d(db::beta_c);
	PrintData(M);
}