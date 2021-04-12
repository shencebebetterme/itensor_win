#pragma once
#include "../inc.h"
#include "itensor/all.h"

void dmrg_ising() {
	int N = 30;
	//int N = std::atoi(argv[1]);
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
	printfln("E/N = %.20f", energy / N);
	//compare with https://dmrg101-tutorial.readthedocs.io/en/latest/tfim.html
}