//Make f2v components datum::nan.
#include<armadillo>
using namespace arma;
struct f2v{double value; vec grad; mat hess;};
f2v nanf2v(const int & order, const f2v & x){
    f2v results;
    results.value=datum::nan;
    return results;
}
