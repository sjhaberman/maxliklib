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
f2v multlogit(const ivec & y, const vec & beta)
{
    double s;
    int r,z;
    vec e;
    r=beta.n_elem;
    f2v results;
    results.grad.set_size(r);
    results.hess.set_size(r,r);
    e=exp(beta);
    s=1.0+sum(e);
    results.value=-log(s);
    results.grad=-e/s;
    results.hess=diagmat(results.grad)+results.grad*trans(results.grad);
    if(y(0)>0)
    {
        z=y(0)-1;
        results.value=beta(z)+results.value;
        results.grad(z)=1.0+results.grad(z);
    }
    return results;
}
