
//#include "inc.h"
//#include "util/database.h"
//#include "util/extract_mat.h"
//#include "util/cft_data.h"
//#include "util/glue.h"
//#include "util/it_help.h"
//
//#include "TRG/tnr.h"
//#include "TRG/trg.h"
//#include "TRG/GiltTNR.h"
//
//#include "opt/descend.h"
//
//#include <chrono>
//using namespace std::chrono;

//ITensor trg(ITensor, int, int);

//#include "sample/ctmrg.h"
//#include "itensor/all.h"
//#include "util/arnoldi.h"
//
#define ARMA_DONT_USE_FORTRAN_HIDDEN_ARGS
#define ARMA_USE_LAPACK

#include "cblas.h"
#include "lapack.h"
#include <armadillo>

int main() {
	arma::mat A = arma::randu<arma::mat>(10, 10);
	//arma::cx_mat B = arma::logmat(A);
	
	std::cout << "hello world" << std::endl;
}