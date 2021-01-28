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








//#include "sample/ctmrg.h"
#include "itensor/all.h"

int main() {
	int N = 6;
	auto sites = SpinHalf(N);
	auto indices = sites.inds();

	auto psi = randomITensor(QN({ "Sz",0 }), indices);

	auto ampo = AutoMPO(sites);
	for (auto j : range1(N - 1))
	{
		ampo += 0.5, "S+", j, "S-", j + 1;
		ampo += 0.5, "S-", j, "S+", j + 1;
		ampo += "Sz", j, "Sz", j + 1;
	}
	//ampo += "Sz", N, "Sz", 1; //periodic boundary condition?
	auto Hmpo = toMPO(ampo);

	auto H = Hmpo(1);
	for (auto j : range1(2, N)) H *= Hmpo(j);

	//Create expH = exp(-tau*H)
	auto expH = expHermitian(H, -8);

	auto gs = psi; //initialize to psi
	int Npass = 4;
	for (int n = 1; n <= Npass; ++n)
	{
		println(n,"-th step");
		gs *= expH;
		gs.noPrime();
		gs /= norm(gs);
	}
	Print(gs);


	//Compute the ground state energy
	auto E0 = elt(prime(dag(gs)) * H * gs);
	Print(E0);


	auto state = InitState(sites);
	for (auto i : range1(N))
	{
		if (i % 2 == 1) state.set(i, "Up");
		else         state.set(i, "Dn");
	}

	auto mps = randomMPS(state);
	auto sweeps = Sweeps(4);
	sweeps.maxdim() = 10, 10, 20, 20;
	sweeps.cutoff() = 1E-10;

	E0 = dmrg(mps, Hmpo, sweeps, { "Quiet",true });
	
	Print(E0);
}

