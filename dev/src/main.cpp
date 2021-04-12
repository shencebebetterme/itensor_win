#include "pch.h"

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





//#include "sample/ctmrg.h"
//#include "itensor/all.h"
//#include "util/arnoldi.h"
// 
//#include "util/arpack_wrap.h"
//#include "sample/arpack_test.h"
//#include "sample/adjoint.h"



template<typename T>
class MyClass
{
public:
	T x_;
	MyClass(T x) {
		x_ = x;
	}


private:

};


template<typename T, typename T2>
T2 f(MyClass<T>& mc_) {
	mc_.x_ = 0;
	T2 z = mc_.x_;
	return z;
}





int main(int argc, char** argv) {

	/*std::vector<int> v1 = { 3,8,7,9 };
	std::vector<std::string> v2 = { "h","e","l","o" };
	ssort(v1, v2);*/

	//arpack_test2();

	//const double beta_c = 0.5 * log(1 + sqrt(2));
	//ITensor A0 = database::ising2d(beta_c);

	////Todo: extract L0 + L0bar from transfer matrix
	//int n_chain = 4;
	//const double c_ising = 0.5;//central charge
	//const double f_ising = 0.929695;//free energy per site
	//const double fA = f_ising * 2; //free energy per A tensor

	//A0 /= fA;
	//ITensor M = glue_bare_ring(A0, n_chain);

	//double factor = std::exp((2 * PI / n_chain) * (c_ising / 12) + n_chain * fA);
	//M /= factor;
	//
	////todo: how to check that M is nearly identity?

	//adtest2();
	//arpack_test2();
	//arpack_test();


	//todo: 1d ising quantum chain
	//int N = 30;
	int N = std::atoi(argv[1]);
	//auto sites = SpinHalf(N);
	auto sites = SpinHalf(N, { "ConserveQNs=",false });

	double h = 1.0; //at critical point

	auto ampo = AutoMPO(sites);

	for (int j = 1; j < N; ++j)
	{
		ampo += -4.0, "Sz", j, "Sz", j + 1;
	}
	for (int j = 1; j <= N; ++j) {
		ampo += -2.0 * h, "Sx", j;
	}

	auto H = toMPO(ampo);
	//auto H = MPO(ampo);

	auto sweeps = Sweeps(5);
	sweeps.maxdim() = 10, 40, 100, 200, 200;
	sweeps.cutoff() = 1E-8;

	// Create a random starting state
	auto state = InitState(sites);
	for (auto i : range1(N))
	{
		if (i % 2 == 1) state.set(i, "Up");
		else         state.set(i, "Dn");
	}
	auto psi0 = randomMPS(state);

	auto [energy, psi] = dmrg(H, psi0, sweeps, { "Quiet",true });

	printf("\n");
	printfln("Ground state energy = %.20f", energy);
	printfln("E/N = %.20f", energy/N);
	//compare with https://dmrg101-tutorial.readthedocs.io/en/latest/tfim.html
}

