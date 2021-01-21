#include "inc.h"
#include "util/database.h"
#include "util/extract_mat.h"
#include "util/cft_data.h"
#include "util/glue.h"
#include "util/it_help.h"

#include "TRG/tnr.h"
#include "TRG/trg.h"
#include "TRG/GiltTNR.h"

#include <chrono>
using namespace std::chrono;

//ITensor trg(ITensor, int, int);


void testCFT(ITensor A) {
	printf("\n testing CFT \n");
	Print(A);
	ITensor As = glue(A, 2);
	Print(As);
	arma::mat Asmat = extract_mat(As);
	//Asmat.print("A");
	printf("\n");
	cd_dense(Asmat, 5, 2, 1);
}


//todo: windows link to hdf5 and openmp
int main() {
	const double beta_c = 0.5 * log(1 + sqrt(2));
	const int n = 1000;
	mat A = randu<mat>(n, n);
	mat B = randu<mat>(n, n);


	auto start = high_resolution_clock::now();
	B = A * B;
	auto stop = high_resolution_clock::now();

	auto duration = duration_cast<microseconds>(stop - start);
	std::cout << duration.count() / 1000000.0 << std::endl;

}

