#include "itensor/all_basic.h"
#include "itensor/util/print_macro.h"

using namespace itensor;
using namespace std;

int main() {
	Index i(3, "i");
	Index j(3, "j");
	ITensor A = randomITensor(i, j);
	//Print(A);
	PrintData(A);
	//std::cout << A << std::endl;

	autovector<double> v(0, 1);
	auto l = v.size();
	auto p1 = v.begin();
	auto p2 = v.end();


	/*auto extractReal = [](Dense<Real> const& d)
	{
		return d.store;
	};

	auto v = applyFunc(extractReal, A.store());*/
}