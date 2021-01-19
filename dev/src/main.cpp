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

int main() {
	const double beta_c = 0.5 * log(1 + sqrt(2));

	/*autovector<double> v(0, 1);
	auto l = v.size();
	auto p1 = v.begin();
	auto p2 = v.end();*/

	ITensor A = database::ising2d(beta_c);
	PrintData(A);
	ITensor As = glue(A, 2);
	PrintData(As);
	arma::mat Asmat = extract_mat(As);
	Asmat.print("Asmat=");
	int k = 1;
	std::cin.get();
}

