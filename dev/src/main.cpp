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
#include "util/arpack_wrap.h"
//#include "sample/arpack_test.h"
#include "sample/adjoint.h"



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





int main() {

	std::vector<int> v1 = { 3,8,7,9 };
	std::vector<std::string> v2 = { "h","e","l","o" };

	ssort(v1, v2);

	//arpack_test2();

	const double beta_c = 0.5 * log(1 + sqrt(2));
	ITensor A0 = database::ising2d(beta_c);

	ITensor B = replaceTags(A0, "d,u", "u,uu");

	//adtest2();
	//arpack_test2();
	//arpack_test();
}

