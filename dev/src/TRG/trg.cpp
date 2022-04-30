#include "pch.h"
#include "trg.h"



ITensor trg_step(ITensor A1, ITensor A2, int maxdim, double cutoff) {
	A1.addTags("A1");
	A2.addTags("A2");

	Index u1 = findIndex(A1, "u");
	Index r1 = findIndex(A1, "r");
	Index d1 = findIndex(A1, "d");
	Index l1 = findIndex(A1, "l");

	Index u2 = findIndex(A2, "u");
	Index r2 = findIndex(A2, "r");
	Index d2 = findIndex(A2, "d");
	Index l2 = findIndex(A2, "l");

	A1.removeTags("Link");
	A2.removeTags("Link");

	//Print(A1); Print(A2);

	auto [NW, SE] = factor(A2, { l2,u2 }, { "MaxDim=",maxdim,"Cutoff=",cutoff, "Tags=","Link,l","ShowEigs=",false,"SVDMethod=","gesdd" });
	NW.replaceTags("l", "r", "Link");
	//Print(NW); Print(SE);

	auto [NE, SW] = factor(A1, { u1,r1 }, { "MaxDim=",maxdim,"Cutoff=",cutoff, "Tags=","Link,u","ShowEigs=",false,"SVDMethod=","gesdd" });
	NE.replaceTags("u", "d", "Link");

	//contraction, just in case the original A1 and A2 have shared indices
	SE.replaceInds({ r2,d2 }, { l1,u1 });
	ITensor A_new = SE * NE * SW;
	A_new *= delta(r1, l2) * NW * delta(d1, u2);

	A_new.removeTags("Link");

	return A_new;
}


// TRG for square 4-leg tensor
// A0 must have indices with tags "u", "r", "d", "l"
ITensor trg(ITensor A, int maxdim, int scale) {
	Index u = findIndex(A, "u");
	Index r = findIndex(A, "r");
	Index d = findIndex(A, "d");
	Index l = findIndex(A, "l");

	bool show_eig = false;

	double fps = 0; //free energy per site

	for (auto s : range1(scale))
	{
		printfln("\n---------- Scale %d -> %d  ----------", s - 1, s);

		// Get the upper-left and lower-right tensors
		auto [Fl, Fr] = factor(A, { r,d }, { l,u }, { "MaxDim=",maxdim,"Tags=","l,scale=" + str(s),"ShowEigs=",show_eig,"SVDMethod=","gesdd" });
		auto l_new = commonIndex(Fl, Fr);
		auto r_new = replaceTags(l_new, "l", "r");
		Fr *= delta(l_new, r_new);

		// Get the upper-right and lower-left tensors
		auto [Fu, Fd] = factor(A, { l,d }, { u,r }, { "MaxDim=",maxdim,"Tags=","u,scale=" + str(s),"ShowEigs=",show_eig,"SVDMethod=","gesdd" });
		auto u_new = commonIndex(Fu, Fd);
		auto d_new = replaceTags(u_new, "u", "d");
		Fd *= delta(u_new, d_new);

		// relabel the indices to contract the 4 F tensors
		// to form the new A tensor
		Fl *= delta(r, l);
		Fu *= delta(d, u);
		Fr *= delta(l, r);
		Fd *= delta(u, d);
		A = Fl * Fu * Fr * Fd;

		//Print(A);

		// Update the indices
		l = l_new;
		r = r_new;
		u = u_new;
		d = d_new;

		// Normalize the current tensor and keep track of
		// the total normalization
		Real TrA = elt(A * delta(l, r) * delta(u, d));
		A /= TrA;
		//PrintData(A);
		fps += 1.0 / pow(2, 1 + s) * log(TrA);
		//Print(A);
	}

	return A;
}

