//Apply linear transformation to f2v components.  order
//indicates what f2v components are used.  Original f2v is x.
//Transformation is transpose of a.
#include<armadillo>
using namespace arma;
struct f2v
{
    double value;
    vec grad;
    mat hess;
};
f2v linsel(const int & order, const f2v & x, const mat & a)
{
    f2v result;
    result.value=x.value;
    if(order>0) result.grad=a.t()*x.grad;
    if(order>1) result.hess=a.t()*x.hess*a;
    return result;
}
