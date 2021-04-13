#pragma once
#include "../inc.h"
#include "itensor/all.h"

void dmrg_ising() {
	int N = 32;
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

	//random initial guess
	auto psi0 = randomMPS(state);

	auto [energy, psi] = dmrg(H, psi0, sweeps, { "Quiet",true });

	printf("\n");
	printfln("Ground state energy = %.20f", energy);
	printfln("E/N = %.20f", energy / N);
	//compare with https://dmrg101-tutorial.readthedocs.io/en/latest/tfim.html
}


void dmrg_ising_excited() {
	int N = 100;
	//int N = std::atoi(argv[1]);
	//auto sites = SpinHalf(N);
	auto sites = SpinHalf(N, { "ConserveQNs=",false });

	double h = 1.0; //at critical point

	auto ampo = AutoMPO(sites);

	for (int j = 1; j < N; ++j)
	{
		ampo += -4.0, "Sz", j, "Sz", j + 1;
	}
	
	ampo += -4.0, "Sz", 1, "Sz", N;

	for (int j = 1; j <= N; ++j) {
		ampo += -2.0 * h, "Sx", j;
	}

	auto H = toMPO(ampo);
	//auto H = MPO(ampo);

	auto sweeps = Sweeps(30);
	sweeps.maxdim() = 10, 40, 100, 200, 200;
	sweeps.cutoff() = 1E-10;
	sweeps.niter() = 2;
	sweeps.noise() = 1E-7, 1E-8, 0.0;


	auto [en0, psi0] = dmrg(H, randomMPS(sites), sweeps, { "Quiet=",true });

	println("\n----------------------\n");

	auto wfs = std::vector<MPS>(1);
	wfs.at(0) = psi0;

	auto [en1, psi1] = dmrg(H, wfs, randomMPS(sites), sweeps, { "Quiet=",true,"Weight=",20.0 });

	println("\n----------------------\n");

	wfs.push_back(psi1);
	auto [en2, psi2] = dmrg(H, wfs, randomMPS(sites), sweeps, { "Quiet=",true,"Weight=",20.0 });

	printfln("\nGround State Energy = %.10f", en0);
	printfln("Ground State Energy / N = %.10f", en0/N);
	printfln("\nExcited State Energy = %.10f", en1);
	printfln("\nDMRG energy gap = %.10f", en1 - en0);
	printfln("\n1st scaling dim gap = %.10f", (en1 - en0) * N / (2 * PI));
	printfln("\n2nd scaling dim gap = %.10f", (en2 - en0) * N / (2 * PI));

	printfln("\nOverlap <psi0|psi1> = %.2E", inner(psi0, psi1));
	printfln("\nOverlap <psi0|psi2> = %.2E", inner(psi0, psi2));
	printfln("\nOverlap <psi1|psi2> = %.2E", inner(psi1, psi2));
}