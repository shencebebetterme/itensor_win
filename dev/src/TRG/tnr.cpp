#include "pch.h"
#include "tnr.h"


// A_ must have indices with tags "u", "r", "d", "l"
TNR::TNR(ITensor& A_) {
	A = A_;
	u = findIndex(A, "u");
	r = findIndex(A, "r");
	d = findIndex(A, "d");
	l = findIndex(A, "l");
}


// svd a tensor, return VU^dagger
ITensor update(const ITensor& AA, const IndexSet& is, const int maxdim) {
	auto [UU, S, V] = svd(AA, is, { "MaxDim=",maxdim, "SVDMethod=","gesdd" });
	Index us = commonIndex(UU, S);
	Index vs = commonIndex(V, S);
	return UU * delta(us, vs) * V;
}


void TNR::buildA4() {
	ITensor A2 = A * delta(r, prime(r)) * prime(A);
	//A2.replaceInds({ prime(l) }, { prime(r) });
	A2.replaceTags("l,1", "r,1");
	//A2.replaceTags("l", "r", "1");
	Print(A2);
	ITensor A2p = prime(conj(A2), 2);
	//Print(A2);

	//todo: check upside-down indices for tensor dagger
	A4 = A2 * delta(d, prime(u, 2));
	A4 *= delta(prime(d), prime(u, 3));
	A4 *= A2p;
}


void TNR::tnr_step() {
	this->buildA4();
	this->build_uvw();
	// TRG step
	this->calculateB();
	this->build_z_and_A();
	//this->calculateC();
	//this->trg_step();
	//A /= norm(A); 

	PrintData(A);
	A /= 1520.3487976;
	PrintData(A);
	//PrintNice("A is", A);
}

void TNR::initial_uvw() {

	Index x = Index(dim(u));
	ul = addTags(x, "ul");
	ur = addTags(x, "ur");
	dl = addTags(x, "dl");
	dr = addTags(x, "dr");
	U = toDense(delta(ul, dl)) * delta(ur, dr);
	Uc = conj(U);

	ITensor L, S1;
	std::tie(vLc, S1, L) = svd(A, { u,l }, { "MaxDim=",chi_tnr, "ShowEigs=",false, "SVDMethod=","gesdd" });
	Print(vLc);
	vLc.replaceTags("Link", "i");
	vLc.replaceTags("l", "o1");
	vLc.replaceTags("u", "o2");
	vLc.removeTags("U,V");
	vLc.addTags("L");
	//
	vL = conj(vLc);
	//
	Li = findIndex(vL, "i");
	Lo1 = findIndex(vL, "o1");
	Lo2 = findIndex(vL, "o2");

	//PrintData(vL * delta(Lo1, prime(Lo1)) * delta(Lo2, prime(Lo2)) * prime(vL));

	if (horz_refl) {
		Print(vL);
		vR = replaceTags(swapTags(vL, "o1", "o2"), "L", "R");
		Print(vR);
		Ri = findIndex(vR, "i");
		Ro1 = findIndex(vR, "o1");
		Ro2 = findIndex(vR, "o2");
		vRc = conj(vR);
	}
	else {
		// already debugged
		ITensor R, S2;
		std::tie(vR, S2, R) = svd(A, { l,d }, { "MaxDim=",chi_tnr, "SVDMethod=","gesdd" });
		vR.replaceTags("Link", "i");
		vR.replaceTags("d", "o1");
		vR.replaceTags("l", "o2");
		vR.removeTags("U,V");
		vR.addTags("R");
		//
		vRc = conj(vR);
		//
		Ri = findIndex(vR, "i");
		Ro1 = findIndex(vR, "o1");
		Ro2 = findIndex(vR, "o2");
	}
}

void TNR::build_uvw() {
	double orig_norm = norm(A4);
	this->initial_uvw();
	this->calculateB();
	double B_norm_old = norm(B);
	double B_norm_change = 100.0;
	//
	int counter = 0;
	int full_dim = dim(Lo1) * dim(Lo2);
	if (full_dim > chi_tnr) {
		printf("\noptimizing isometries and disentangler ...\n");
		while ((counter < max_iter) && (B_norm_change > norm_change_bound))
		{
			counter += 1;

			this->optimize_isometries();
			//PrintData(vL);
			this->optimize_disentangler();
			//nB = B_norm;
			B_norm_change = abs(B_norm - B_norm_old) / B_norm_old;
			B_norm_old = B_norm;
		}
		Print(counter);
		Print(B_norm_change);
	}
}

// update vL, vR, vLc, vRc and related indices
// use doubled version
void TNR::optimize_isometries() {
	this->calculateB_lower();
	ITensor Atmp = swapTags(A, "l", "r");
	Print(norm(A - Atmp));
	ITensor A2 = A * delta(r, prime(l)) * prime(A);
	//ITensor A2p = prime(conj(A2), 2);
	Print(A2); Print(U); Print(vR);
	ITensor A2U = replaceInds(A2, { u,prime(u),prime(r) }, { dl,dr,Ro1 }) * U;
	Print(A2U);
	ITensor env_top = A2U * delta(ur, Ro2) * vR;
	Print(env_top);
	ITensor A2vR = A2 * delta(prime(r), Ro1) * vR * delta(prime(u), Ro2);
	Print(A2vR);
	//PrintData(env_top);

	ITensor half_GvL = replaceInds(B_lower, { u,prime(u) }, { dl,dr }) * U;
	half_GvL.replaceInds({ ur,prime(r) }, { Ro2,Ro1 });
	half_GvL *= vR;
	//PrintData(half_GvL);
	ITensor GvL = half_GvL * prime(conj(half_GvL), { l,ul });
	//Print(GvL);
	//PrintData(GvL);
	std::tie(vL, tmp) = diagPosSemiDef(GvL, { "MaxDim=",chi_tnr });
	//Print(vL);
	vL.replaceTags("Link", "i");
	vL.replaceTags("l", "o1");
	vL.replaceTags("ul", "o2");
	vL.removeTags("U,V");
	vL.addTags("L");
	//
	vLc = conj(vL);
	Li = findIndex(vL, "i");
	Lo1 = findIndex(vL, "o1");
	Lo2 = findIndex(vL, "o2");
	//
	if (horz_refl) {
		vR = replaceTags(vL, "L", "R");
		vRc = conj(vR);
		Ri = findIndex(vR, "i");
		Ro1 = findIndex(vR, "o1");
		Ro2 = findIndex(vR, "o2");
	}
}

// update U, Uc and related indices
// calculate B_norm
void TNR::optimize_disentangler() {
	this->calculateB();
	BBl = Bc * B_lower;
	//Print(BBl);

	// calculate environments
	Gamma_U = BBl * prime(vL, 3, Li);
	Gamma_U *= delta(Lo1, l);
	Gamma_U *= prime(vR, 2, Ri);
	Gamma_U *= delta(Ro1, prime(r));

	// update tensors
	U = update(Gamma_U, findInds(Gamma_U, "u"), chi_tnr);

	// update B_norm
	B_norm = sqrt(eltC(U * Gamma_U).real());

	U.replaceTags("o2,L", "ul");
	U.replaceTags("o2,R", "ur");
	U.replaceTags("u,0", "dl,0");
	U.replaceTags("u,1", "dr,0");
	//
	ul = findIndex(U, "ul");
	ur = findIndex(U, "ur");
	dl = findIndex(U, "dl");
	dr = findIndex(U, "dr");
	//symmetrize U to kill numerical errors
	ITensor U_dag = swapInds(U, { ul,dl }, { ur,dr }).conj();
	U = (U + U_dag) / 2;
	//U.setPrime(0);
	//
	Uc = conj(U);
}



void TNR::calculateB_lower() {
	// B_lower
	B_lower = A4 * delta(prime(d, 2), dl) * delta(prime(d, 3), dr);
	//Print(B_lower);
	B_lower *= Uc;
	//
	B_lower *= delta(prime(l, 2), Lo1);
	B_lower *= delta(ul, Lo2);
	B_lower *= vLc;
	B_lower *= delta(prime(r, 3), Ro1);
	B_lower *= delta(ur, Ro2);
	B_lower *= prime(vRc, Ri); // to distinguish from vLc "i"
	//Print(B_lower);
}

void TNR::calculateB() {

	this->calculateB_lower();
	// B
	B = B_lower * delta(u, dl) * delta(prime(u), dr);
	B *= U;
	B *= delta(l, Lo1);
	B *= delta(ul, Lo2);
	B *= prime(vL, 3, Li);
	B *= delta(prime(r), Ro1);
	B *= delta(ur, Ro2);
	B *= prime(vR, 2, Ri);
	// now B has 4 indices with tags i, i', i'', i'''
	//Print(B);
	Bc = conj(B);
}

void TNR::calculateC() {
	//C = prime(vR) * delta(Ro1, prime(Ro1)) * vRc;
	//C *= delta(Ro2, prime(Lo2));
	//C *= prime(vLc);
	//C *= delta(prime(Lo1), Lo1);
	//C *= delta(prime(Ro2), Lo2);
	//C *= vL;
	ITensor Cd = vRc * delta(Ro1, Lo1) * prime(vLc, Li);
	ITensor Cu = prime(vR, Ri) * delta(Ro1, Lo1) * vL;
	C = Cd * Cu;

	//Print(C);
}


// deprecated
// 
// u must have indices with tags "ul", "ur", "dl", "dr"
// vL and vR must indices with tags "i", "o1", "o2"
// the above tensors should not have other tags
void TNR::optimize_uvw() {
	calculateB();
	BBl = Bc * B_lower;
	//Print(BBl);

	// calculate environments
	Gamma_U = BBl * prime(vL, 3, Li);
	Gamma_U *= delta(Lo1, l);
	Gamma_U *= prime(vR, 2, Ri);
	Gamma_U *= delta(Ro1, prime(r));
	//Print(Gamma_U);

	// environments of vL and vR
	BBlU = BBl * delta(u, dl);
	BBlU *= delta(prime(u), dr);
	BBlU *= U;
	//Print(BBlU);
	Gamma_vL = BBlU * prime(vR, 2, Ri);
	Gamma_vL *= delta(prime(r), Ro1);
	Gamma_vL *= delta(ur, Ro2);
	//Print(Gamma_vL);
	Gamma_vR = BBlU * prime(vL, 3, Li);
	Gamma_vR *= delta(l, Lo1);
	Gamma_vR *= delta(ul, Lo2);
	//Print(Gamma_vR);


	// update tensors
	U = update(Gamma_U, findInds(Gamma_U, "u"), chi_tnr);
	U.replaceTags("o2,L", "ul");
	U.replaceTags("o2,R", "ur");
	U.replaceTags("u,0", "dl,0");
	U.replaceTags("u,1", "dr,0");
	//
	ul = findIndex(U, "ul");
	ur = findIndex(U, "ur");
	dl = findIndex(U, "dl");
	dr = findIndex(U, "dr");
	//U.setPrime(0);
	//
	Uc = conj(U);

	vL = update(Gamma_vL, { prime(Li,3) }, chi_tnr);
	vL.setPrime(0);
	vL.replaceTags("l", "o1");
	vL.replaceTags("ul", "o2");
	vL.addTags("L");
	//
	Li = findIndex(vL, "i");
	Lo1 = findIndex(vL, "o1");
	Lo2 = findIndex(vL, "o2");
	//
	vLc = conj(vL);

	vR = update(Gamma_vR, { prime(Ri,2) }, chi_tnr);
	vR.setPrime(0);
	vR.replaceTags("r", "o1");
	vR.replaceTags("ur", "o2");
	vR.addTags("R");
	//
	Ri = findIndex(vR, "i");
	Ro1 = findIndex(vR, "o1");
	Ro2 = findIndex(vR, "o2");
	//
	vRc = conj(vR);
}


void TNR::trg_step() {
	//std::cout << "\n trg step\n";
	//IndexSet Bis = { findIndex(B,"i,0"),findIndex(B,"i,3") };
	auto [BL, BR] = factor(B, findInds(B, "L"), { "MaxDim=",chi_trg_B,"Cutoff=",1E-6,"Tags=","l","ShowEigs=",true,"SVDMethod=","gesdd" });
	BL.replaceTags("l", "r");
	//PrintData(BL);
	//PrintData(BR);

	//PrintData(C);
	IndexSet Cis = { findIndex(C,"L,i,0"),findIndex(C,"R,i,1") };
	auto [CU, CD] = factor(C, Cis, { "MaxDim=", chi_trg_C, "Cutoff=",1E-6, "Tags=","u","ShowEigs=",true,"SVDMethod=","gesdd" });
	CU.replaceTags("u", "d");
	//PrintData(CU);
	//PrintData(CD);

	BL.removeTags("U,V,Link");
	BR.removeTags("U,V,Link");
	CU.removeTags("U,V,Link");
	CD.removeTags("U,V,Link");

	//Print(BL); Print(BR); Print(CU); Print(CD);

	A = BR * CU * BL * prime(CD, 2);
	A.setPrime(0);
	//PrintData(A);
	//
	u = findIndex(A, "u");
	r = findIndex(A, "r");
	d = findIndex(A, "d");
	l = findIndex(A, "l");

}


void TNR::build_z_and_A() {
	IndexSet Bis = { findIndex(B,"i,0"),findIndex(B,"i,3") };
	auto [BL, BR] = factor(B, Bis, { "MaxDim=",chi_trg_B,"Tags=","l","ShowEigs=",true,"SVDMethod=","gesdd" });
	//
	//PrintData(BL); PrintData(BR); PrintData(vL); PrintData(vR);
	BL.replaceTags("l", "r");

	/*
	ITensor vLcvRc = vLc * delta(Lo2, Ro2) * vRc;
	Print(vLcvRc);
	ITensor vLcvRc_BR = vLcvRc * delta(Li, prime(Ri, 2)) * BR;
	Print(vLcvRc_BR);
	ITensor vLvR = vR * delta(Ro2, Lo2) * vL;
	Print(vLvR);
	ITensor vLvR_BL = vLvR * BL;
	Print(vLvR_BL);
	ITensor hGz = prime(vLvR_BL,Ri) * delta(prime(Li, 3), Ri) * prime(vLcvRc_BR, {Lo1,Ro1});
	//hGz *= delta(Ri, prime(Ri));
	Print(hGz);
	*/

	ITensor vLvR = vL * delta(Lo1, Ro1) * vR;
	Print(vLvR);
	//ITensor vLcvRc = conj(vLvR);
	ITensor halfGz = BR * delta(prime(Ri), Ri) * vLvR * BL;
	halfGz.replaceInds({ prime(Ri,2),prime(Li,3) }, { prime(Ri),prime(Li) });
	halfGz *= prime(conj(vLvR));
	Print(halfGz);

	//ITensor halfGz = BR * delta(prime(Ri), Ri) * vR;
	//halfGz *= delta(Ro2, Lo2) * vL * BL;
	//halfGz *= delta(prime(Ri,2), Ri) * prime(vRc, Ro1);
	////Print(halfGz); Print(BL);
	////halfGz *= BL;
	//halfGz *= delta(Ro2, Lo2);
	//halfGz *= delta(prime(Li,3), Li) * prime(vLc, Lo1);
	//Print(halfGz);

	ITensor halfGz_copy = halfGz;
	halfGz.addTags("t", "1");//"t" is the temporary tag
	ITensor Gz = halfGz * conj(halfGz_copy);
	Gz.replaceTags("1", "0", "t");//unprime indices with tag "t"
	Gz.removeTags("t");
	halfGz.removeTags("t");
	//Print(Gz);

	std::tie(z, tmp) = diagPosSemiDef(Gz, { "MaxDim=",chi_trg,"ShowEigs=",true });
	//std::tie(z, tmp) = diagHermitian(Gz);
	//PrintData(tmp);
	//PrintData(z);

	Print(z);
	A = halfGz * conj(z) * prime(z);
	A.replaceTags("Link,0", "d,0");
	A.replaceTags("Link,1", "u,0");
	Print(A);
	u = findIndex(A, "u");
	r = findIndex(A, "r");
	d = findIndex(A, "d");
	l = findIndex(A, "l");
	Print(A);
}

