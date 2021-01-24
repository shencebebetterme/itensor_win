#include "inc.h"
#include "util/database.h"
#include "util/extract_mat.h"
#include "util/cft_data.h"
#include "util/glue.h"
#include "util/it_help.h"

#include "TRG/tnr.h"
#include "TRG/trg.h"
#include "TRG/GiltTNR.h"

#include "opt/descend.h"

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
	ITensor A0 = database::ising2d(beta_c);
	const double c_ising = 0.5;//central charge
	const double f_ising = 0.929695;//free energy per site
	const double fA = f_ising * 2; //free energy per A tensor
	
	constexpr int n_chain = 4;
	ITensor MT = glue(A0, n_chain, true, false);
	//Print(MT);
	double factor = std::exp((2 * PI / n_chain) * (c_ising / 12) + n_chain * fA);
	MT /= factor; // remove central charge and free energy
	ITensor logMT = tensor_log(MT);

	//obtain L0+L0bar and L0-L0bar
	ITensor sum = -(n_chain / (2 * PI)) * realPart(logMT);
	ITensor diff = (n_chain / (2 * PI)) * imagPart(logMT);
	//PrintData(sum); PrintData(diff);
	ITensor L0 = (sum - diff) / 2;
	Print(L0);

	ITensor Lm1 = L0; //L-1
	//apply rotation matrix to down legs according to prime level
	IndexSet Uis = findInds(L0, "u");
	IndexSet Dis = findInds(L0, "d");
	for (auto id : Dis) {
		double theta = 2 * PI * primeLevel(id) / n_chain;
		double cos_theta = std::cos(theta);
		double sin_theta = std::sin(theta);

		Index idp = addTags(id, "ex");
		ITensor gi(id, idp);
		// apply gauge
		if (dim(id) == 2 ) {
			gi.set(id = 1, idp = 1, cos_theta);
			gi.set(id = 1, idp = 2, -sin_theta);
			gi.set(id = 2, idp = 1, sin_theta);
			gi.set(id = 2, idp = 2, cos_theta);
		}
		Lm1 *= gi;
	}
	Lm1.removeTags("ex");

	//todo: change L0 and L1 tensor into matrix
	auto [uT, Uidx] = combiner(Uis);
	auto [dT, Didx] = combiner(Dis);
	L0 = uT * L0 * dT;
	Lm1 = uT * Lm1 * dT;
	auto L0mat = extract_cxmat(L0);
	auto Lm1mat = extract_cxmat(Lm1);
	L0mat.print("L0mat"); Lm1mat.print("L1mat");

	arma::cx_mat comm = Lm1mat * L0mat - L0mat * Lm1mat;
	(comm - Lm1mat).print("diff");

	//todo: enable debug mode hasIndex() and error stack
}

