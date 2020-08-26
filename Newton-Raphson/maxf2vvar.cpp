//Combine argument y and corresponding function, gradient, and definition
//function in vary into a single entity maxf1vvar.
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
maxf2v maxf2vvar(const vec &y,const f2v & fy)
{
    maxf2v result;
    result.locmax=y;
    result.max=fy.value;
    result.grad=fy.grad;
    result.hess=fy.hess;
    return result;
}

