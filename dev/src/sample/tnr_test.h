#pragma once

int tnr_test() {
	ITensor A = database::ising2d(0.5);

	Index u = findIndex(A, "u");
	Index r = findIndex(A, "r");
	Index d = findIndex(A, "d");
	Index l = findIndex(A, "l");

	ITensor U = A;
	U.replaceTags("u", "ul");
	U.replaceTags("r", "ur");
	U.replaceTags("d", "dr");
	U.replaceTags("l", "dl");

	Index i(2, "i");
	Index o1(2, "o1");
	Index o2(2, "o2");

	ITensor vL = randomITensor(i, o1, o2);
	ITensor vR = randomITensor(i, o1, o2);

	optimize_u_vL_vR(A, U, vL, vR);
}