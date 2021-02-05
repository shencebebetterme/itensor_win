#include "pch.h"
#include "arnoldi.h"



int find2ndEig(Vector const & vr, Vector const & vi) {
	int n1 = -1;
	int n2 = -1;
	Real largest_val = NAN;
	Real s_l_val = NAN;
	for (size_t i = 0; i < vr.size(); i++)
	{

			auto ival = abs(Complex(vr(i), vi(i)));
			if (i == 0)
			{
				largest_val = ival;
				s_l_val = largest_val;
				n1 = 0;
				n2 = 0;
			}
			else if (ival >= largest_val)
			{
				s_l_val = largest_val;
				largest_val = ival;
				n2 = n1;
				n1 = i;
			}
			else if (ival < largest_val && i == 1) {
				s_l_val = ival;
				n2 = i;
			}
			else if (ival >= s_l_val && ival < largest_val)
			{
				s_l_val = ival;
				n2 = i;
			}
		
	}
	return n2;
}




// phi will be used to store the eigenvectors
std::vector<Complex>
my_arnoldi(const ITensorMap& A,
	std::vector<ITensor>& phi,
	Args const& args)
{
	//////////////////////////////////////////////////////////////////////////////////
	// set up parameters
	//////////////////////////////////////////////////////////////////////////////////
	int maxiter_ = args.getInt("MaxIter", 10);
	int maxrestart_ = args.getInt("MaxRestart", 0);
	std::string whicheig_ = args.getString("WhichEig", "LargestMagnitude");
	const Real errgoal_ = args.getReal("ErrGoal", 1E-6);
	const int debug_level_ = args.getInt("DebugLevel", -1);
	bool is_Hermitian_ = args.getBool("IsHermitian", false);

	if (maxiter_ < 1) maxiter_ = 1;
	if (maxrestart_ < 0) maxrestart_ = 0;

	const Real Approx0 = 1E-12;
	const int Npass = args.getInt("Npass", 2); // number of Gram-Schmidt passes

	const size_t nget = phi.size();
	if (nget == 0) Error("No initial vectors passed to arnoldi.");

	//if(nget > 1) Error("arnoldi currently only supports nget == 1");


	//////////////////////////////////////////////////////////////////////////////////
	// initial unit vectors
	//////////////////////////////////////////////////////////////////////////////////
	for (size_t j = 0; j < nget; ++j)
	{
		phi[j].randomize();
		const Real nrm = norm(phi[j]);
		if (nrm == 0.0)
			Error("norm of 0 in arnoldi");
		phi[j] *= 1.0 / nrm;
	}


	// eigs will be used to store the eigenvalues
	std::vector<Complex> eigs(nget);

	const int maxsize = A.size(); // matrix size

	if (phi.size() > size_t(maxsize))
		Error("arnoldi: requested more eigenvectors (phi.size()) than size of matrix (A.size())");

	//checked if A is 1x1 matrix
	if (maxsize == 1)
	{
		if (norm(phi.front()) == 0) randomize(phi.front());
		phi.front() /= norm(phi.front());
		ITensor Aphi(phi.front());
		A.product(phi.front(), Aphi);
		//eigs.front() = BraKet(Aphi,phi.front());
		gmres_details::dot(Aphi, phi.front(), eigs.front());
		return eigs;
	}

	auto actual_maxiter = std::min(maxiter_, maxsize - 1);

	if (debug_level_ >= 2)
	{
		printfln("maxsize-1 = %d, maxiter = %d, actual_maxiter = %d",
			(maxsize - 1), maxiter_, actual_maxiter);
	}

	if (dim(phi.front().inds()) != size_t(maxsize))
	{
		Error("arnoldi: size of initial vector should match linear matrix size");
	}

	//Storage for Matrix that gets diagonalized 
	// real and imaginary parts of the Hessenberg matrix
	Matrix HR(actual_maxiter + 2, actual_maxiter + 2),
		HI(actual_maxiter + 2, actual_maxiter + 2);
	//HR = 0;
	//HI = 0;
	for (auto& el : HR) el = 0;
	for (auto& el : HI) el = 0;

	std::vector<ITensor> V(actual_maxiter + 2);




	for (size_t w = 0; w < nget; ++w)
	{
		std::cout << "\n\n==========================================\n";
		std::cout << "w = " << w << std::endl;
		// implicitly restarted arnoldi
		for (int r = 0; r <= maxrestart_; ++r)
		{
			Real err = 1000;
			Matrix YR, YI; // store the eigenvectors
			int n = 0; //which column of Y holds the w^th eigenvector
			//int n2 = 0; // which column holds the second largest eigenvector
			int niter = 0;

			//Mref holds current projection of A into V's
			MatrixRef HrefR(subMatrix(HR, 0, 1, 0, 1)),
				HrefI(subMatrix(HI, 0, 1, 0, 1));

			V.at(0) = phi.at(w);

			////////////////////////////////////////////////////////////
			// arnoldi iteration
			////////////////////////////////////////////////////////////
			for (int it = 0; it <= actual_maxiter; ++it)
			{
				const int j = it;
				A.product(V.at(j), V.at(j + 1)); // V[j+1] = A*V[j]
					// "Deflate" previous eigenpairs:
				for (size_t o = 0; o < w; ++o)
				{
					//V[j+1] += (-eigs.at(o)*phi[o]*BraKet(phi[o],V[j+1]));
					Complex overlap_;
					//gmres_details::dot(phi[o], V[j + 1], overlap_);
					gmres_details::dot(phi[o], V[j], overlap_);
					//auto diff = -eigs.at(o) * phi[o] * overlap_;
					V[j + 1] += -eigs.at(o) * phi[o] * overlap_;
					//V[j + 1].randomize();
				}

			//Do Gram-Schmidt orthogonalization Npass times
			//Build H matrix only on the first pass
				Real nh = NAN;
				for (int pass = 1; pass <= Npass; ++pass)
				{
					for (int i = 0; i <= j; ++i)
					{
						//Complex h = BraKet(V.at(i),V.at(j+1));
						Complex h;
						gmres_details::dot(V.at(i), V.at(j + 1), h);
						//gmres_details::dot(Vcopy.at(i), V.at(j + 1), h);
						if (pass == 1)
						{
							//if (i > w && j > w) {
								HR(i, j) = h.real();
								HI(i, j) = h.imag();
							//}
						}
						V.at(j + 1) -= h * V.at(i);
					}
					Real nrm = norm(V.at(j + 1));
					if (pass == 1) nh = nrm;

					if (nrm != 0) V.at(j + 1) /= nrm;
					else         randomize(V.at(j + 1));
				}

				//for(int i1 = 0; i1 <= j+1; ++i1)
				//for(int i2 = 0; i2 <= j+1; ++i2)
				//    {
				//    auto olap = BraKet(V.at(i1),V.at(i2)).real();
				//    if(fabs(olap) > 1E-12)
				//        Cout << Format(" %.2E") % BraKet(V.at(i1),V.at(i2)).real();
				//    }
				//Cout << Endl;

				//Diagonalize projected form of A to
				//obtain the w^th eigenvalue and eigenvector


					//store the real and imaginary part of eigenvalues
				Vector D(1 + j), DI(1 + j);

				//TODO: eigen only takes a Matrix of Complex, not
				//the real and imaginary parts seperately.
				//Change it so that we don't have to allocate this
				//Complex matrix
				auto Hnrows = nrows(HrefR);
				auto Hncols = ncols(HrefR);
				CMatrix Href(Hnrows, Hncols);
				for (size_t irows = 0; irows < Hnrows; irows++)
					for (size_t icols = 0; icols < Hncols; icols++)
						Href(irows, icols) = Complex(HrefR(irows, icols), HrefI(irows, icols));

				eigen(Href, YR, YI, D, DI);
				n = findEig(D, DI, whicheig_); //continue to target the largest eig 
											//since we have 'deflated' the previous ones
				//n2 = find2ndEig(D, DI);
				eigs.at(w) = Complex(D(n), DI(n));


				HrefR = subMatrix(HR, 0, j + 2, 0, j + 2);
				HrefI = subMatrix(HI, 0, j + 2, 0, j + 2);

				HR(1 + j, j) = nh;

				//Estimate error || (A-l_j*I)*p_j || = h_{j+1,j}*[last entry of Y_j]
				//See http://web.eecs.utk.edu/~dongarra/etemplates/node216.html
				// error of Ritz pair = norm(residual) * last element of eigenvector
				assert(nrows(YR) == size_t(1 + j));
				err = nh * abs(Complex(YR(j, n), YI(j, n)));
				assert(err >= 0);

				if (debug_level_ >= 1)
				{
					if (r == 0)
						printf("I %d e %.0E E", (1 + j), err);
					else
						printf("R %d I %d e %.0E E", r, (1 + j), err);

					for (size_t j = 0; j <= w; ++j)
					{
						if (fabs(eigs[j].real()) > 1E-6)
						{
							if (fabs(eigs[j].imag()) > Approx0)
								printf(" (%.10f,%.10f)", eigs[j].real(), eigs[j].imag());
							else
								printf(" %.10f", eigs[j].real());
						}
						else
						{
							if (fabs(eigs[j].imag()) > Approx0)
								printf(" (%.5E,%.5E)", eigs[j].real(), eigs[j].imag());
							else
								printf(" %.5E", eigs[j].real());
						}
					}
					println();
				}

				++niter;

				if (err < errgoal_) {
					std::cout << "\n for loop over j broken at step " << j << "\n";
					break;
				}

				if (j == actual_maxiter) {
					std::cout << "actual_maxiter reached, error is " << err << std::endl;
				}

			} // for loop over j

		//Cout << Endl;
		//for(int i = 0; i < niter; ++i)
		//for(int j = 0; j < niter; ++j)
		//    Cout << Format("<V[%d]|V[%d]> = %.5E") % i % j % BraKet(V.at(i),V.at(j)) << Endl;
		//Cout << Endl;

		//Compute w^th eigenvector of A
		//Cout << Format("Computing eigenvector %d") % w << Endl;
		 
			//done: make the eigenvectors correct
			//phi.at(w) = Complex(YR(0, n), YI(0, n)) * V.at(0);
			//for (int j = 1; j < niter; ++j)
			//{
			//	phi.at(w) += Complex(YR(j, n), YI(j, n)) * V.at(j);
			//}

			// phi.at(w) = psi - sum_i x_i * phi[i]
			ITensor psi = Complex(YR(0, n), YI(0, n)) * V.at(0);
			for (int j = 1; j < niter; ++j)
			{
				psi += Complex(YR(j, n), YI(j, n)) * V.at(j);
			}

			if (w == 0) phi.at(w) = psi;
			else if (w > 0)
			{
				if (is_Hermitian_) phi.at(w) = psi;
				else {
					// construct theta vector
					std::vector<Complex> theta = {};
					theta.reserve(w);

					for (int j = 0; j < w; j++) {
						Complex z1, z2, s = 0;
						for (int k = 0; k < w; k++) {
							gmres_details::dot(phi[j], phi[k], z1);
							gmres_details::dot(phi[k], psi, z2);
							s += eigs[k] * z1 * z2;
						}
						theta.push_back(s);
					}


					// construct matrix B
					Index j(w, "j");
					Index i(w, "i");
					ITensor B(j, i);
					B.set(j = 1, i = 1, Complex(0, 1.0)); // stupid way to make B has complex elements
					for (int sj = 0; sj < w; sj++) {
						Complex z = 0;
						for (int si = 0; si < w; si++) {
							gmres_details::dot(phi[sj], phi[si], z);
							z *= (eigs[si] - eigs[w]);
							B.set(j = sj + 1, i = si + 1, z);
						}
					}

					//PrintData(B);
					//todo: in the case where some eigenvalues are all degenerate, replace inv(B) by some other matrix

					// calculate B^-1 * theta
					arma::cx_mat Bmat = extract_cxmat(B, false);
					arma::cx_vec theta_vec(theta);
					//std::cout << "when r=" << r << ", det(B)=" << arma::det(Bmat) << std::endl;
					//arma::cx_vec x_vec = arma::inv(Bmat) * theta_vec;
					arma::cx_vec x_vec = solve(Bmat, theta_vec);

					// reconstruct eigs.at(w)
					phi.at(w) = psi;
					for (auto i = 0; i < w; i++) {
						phi.at(w) -= x_vec[i] * phi.at(i);
					}
				}
			}

			//Print(YR.Column(1+n));
			//Print(YI.Column(1+n));

			const Real nrm = norm(phi.at(w));
			if (nrm != 0)
				phi.at(w) /= nrm;
			else
				randomize(phi.at(w));


			//todo: speed up subsequent convergence
			//if (err < errgoal_ || r == maxrestart_) {
			//	// will leave the r loop and w -> w+1
			//	if (w+1 < nget) {
			//		ITensor& pwp = phi.at(w + 1);
			//		pwp = Complex(YR(0, n2), YI(0, n2)) * V.at(0);
			//		for (int j = 1; j < niter; ++j)
			//		{
			//			pwp += Complex(YR(j, n2), YI(j, n2)) * V.at(j);
			//		}
			//		const double nrm = norm(pwp);
			//		if (nrm != 0) pwp /= nrm;
			//	}
			//}

			if (err < errgoal_) {
				std::cout << "\n for loop over r broken at step " << r << "\n";
				break;
			}
			//otherwise restart using the phi.at(w) computed above

			if (r == maxrestart_) {
				std::cout << "\n maxrestart_ reached, error is " << err << "\n\n\n";
			}
		} // for loop over r

		//checking eigenvalue and eigenvectors
		ITensor Av;
		A.product(phi[w], Av);
		std::cout << "checking eigenpair: " << norm(Av - eigs[w] * phi[w]) << std::endl;
	} // for loop over w

	return eigs;
}



Complex
my_arnoldi(const ITensorMap& A,
	ITensor& vec,
	Args const& args = Args::global())
{
	std::vector<ITensor> phi(1, vec);
	Complex res = my_arnoldi(A, phi, args).front();
	vec = phi.front();
	return res;
}



void test_arnoldi_single() {
	//auto i = Index(2, "i");
//auto j = Index(2, "j");
//auto k = Index(2, "k");
//auto A = randomITensor(i, j, k, prime(i), prime(j), prime(k));

	auto i = Index(5, "i");
	auto i1 = prime(i);
	//auto A = randomITensor(i, i1);
	ITensor A(i, i1);
	double val = 1.2;
	for (auto s : range1(i.dim()))
		for (auto t : range1(i1.dim()))
		{

			if (s == t) {
				A.set(s, t, val);
				val -= 0.1;
			}
			else {
				A.set(s, t, 0);
			}
		}
	A += 0.005 * randomITensor(i, i1);
	PrintData(A);

	// Using the ITensorMap class
	// to make an object that can be passed
	// to arnoldi
	auto AM = ITensorMap(A);

	PrintData(AM.size());

	// Random starting vector for Arnoldi
	//auto x = randomITensor(i, j, k);
	auto x = randomITensor(i);

	auto lambda = my_arnoldi(AM, x, { "ErrGoal=",1E-14,"MaxIter=",20,"MaxRestart=",5 });

	// Check it is an eigenvector
	PrintData(lambda);
	PrintData(norm((A * x).noPrime() - lambda * x));
}


void test_arnoldi_multi() {
	auto i = Index(2, "i");
	auto j = Index(2, "j");
	auto k = Index(2, "k");
	auto A = randomITensor(i, j, k, prime(i), prime(j), prime(k));
	//auto A = randomITensor(i, prime(i));
	//A.set(1, 1, 1.0);
	//A.set(1, 2, 2.0);
	//A.set(2, 1, 3.0);
	//A.set(2, 2, 4.0);
	//PrintData(A);

	auto [U, D] = eigen(A);
	PrintData(D);
	//PrintData(U);
	//PrintData(A*U-prime(U)*D);
	//PrintData(prime(U) * D * U);

	//ITensor A2 = A * prime(A); //square of A
	//A2.replaceTags("2", "1");
	//auto [U2, D2] = eigen(A2);
	//PrintData(D2);

	// Using the ITensorMap class
	// to make an object that can be passed
	// to arnoldi
	auto AM = ITensorMap(A);

	PrintData(AM.size());

	// Random starting vector for Arnoldi
	auto x1 = randomITensor(i, j, k);
	//auto x2 = randomITensor(i, j, k);
	//auto x1 = randomITensor(i);
	//auto x2 = randomITensor(i);
	constexpr int nvec = 4;
	std::vector<ITensor> xvec(nvec, x1);

	auto lambda = my_arnoldi(AM, xvec, { "ErrGoal=",1E-4,"MaxIter=",20,"MaxRestart=",5,"Npass=",2 });

	// Check it is an eigenvector
	for (auto i : range(nvec)) {
		std::cout << lambda[i] << "\n";
	}
	//PrintData(xvec[0]); PrintData(xvec[1]);
	PrintData(norm((A * xvec[0]).noPrime() - lambda[0] * xvec[0]));
	//PrintData(norm((A * xvec[1]).noPrime() - lambda[1] * xvec[1]));
}

