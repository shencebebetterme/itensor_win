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


#include "sample/arpack_test.h"

int main() {

	//MyClass mc(3.0);
	//auto a = f<double,double>(mc);

	arpack_test();

}

