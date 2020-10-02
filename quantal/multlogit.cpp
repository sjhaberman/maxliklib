//Log likelihood component, gradient, and hessian matrix
//for multinomial logit model with response vector y with integer values
//from 0 to r-1 and parameter
//vector beta of dimension r-1.
#include<armadillo>
using namespace arma;
struct f2v
{
    double value;
    vec grad;
    mat hess;
};
f2v multlogit(ivec & y,vec & beta)
{
    double r;
    int z;
    vec e;
    f2v results;
    e=exp(beta);
    r=1.0+sum(e);
    results.value=-log(r);
    results.grad=-e/r;
    results.hess=diagmat(results.grad)+results.grad*trans(results.grad);
    if(y(0)>0)
    {
        z=y(0)-1;
        results.value=beta(z)+results.value;
        results.grad(z)=1.0+results.grad(z);
    }
    return results;
}
