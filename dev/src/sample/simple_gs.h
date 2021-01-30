#pragma once

#include "../inc.h"
#include "itensor/all.h"

void simple_gs() {
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

	// imaginary time evolution
	//Create expH = exp(-tau*H)
	auto expH = expHermitian(H, -10);

	auto gs = psi; //initialize to psi
	int Npass = 4;
	for (int n = 1; n <= Npass; ++n)
	{
		gs *= expH;
		gs.noPrime();
		gs /= norm(gs);
	}
	Print(gs);


	//Compute the ground state energy
	auto E0 = elt(prime(dag(gs)) * H * gs);
	Print(E0);

	//Compute the variance <H^2>-<H>^2 to check that gs is
	//an eigenstate. The result "var" should be very small.
	auto H2 = multSiteOps(H, H);
	auto var = elt(prime(dag(gs)) * H2 * gs) - (E0 * E0);
	Print(var);
}


