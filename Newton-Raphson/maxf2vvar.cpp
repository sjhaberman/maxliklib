//Combine argument y and corresponding function, gradient, and hessian
//into a single entity maxf2vvar.
#include<armadillo>
using namespace arma;
struct f2v
{
    double value;
    vec grad;
    mat hess;
};
struct maxf2v
{
    vec locmax;
    double max;
    vec grad;
    mat hess;
};
maxf2v maxf2vvar(const int & order , const vec & y, const f2v & fy)
{
    maxf2v result;
    result.locmax=y;
    result.max=fy.value;
    if(order>0) result.grad=fy.grad;
    if(order>1) result.hess=fy.hess;
    return result;
}

