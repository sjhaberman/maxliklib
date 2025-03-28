//Log likelihood component, gradient, and hessian matrix
//for multinomial logit model with response vector y with integer values
//from 0 to r-1 and parameter
//vector beta of dimension r-1.
//If order is 0, only the function is
//found, if order is 1, then the function and gradient are found.
//If order is 2,
//then the function, gradient, and Hessian are returned.
#include<armadillo>
using namespace arma;
struct f2v
{
    double value;
    vec grad;
    mat hess;
};
struct resp
{
    ivec iresp;
    vec dresp;
};
f2v multlogit(const int & order, const resp & y, const vec & beta)
{
    double s;
    int i, r,z;
    vec e;
    r=beta.n_elem;
    f2v results;
    if(order>0) results.grad.set_size(r);
    if(order>1) results.hess.set_size(r,r);
    e.set_size(r);
    for(i=0;i<r;i++)
    {
        e(i)=exp(beta(i));
    }
    s=1.0+sum(e);
    results.value=-log(s);
    if(order>0) results.grad=-e/s;
    if(order>1) results.hess
        =diagmat(results.grad)+results.grad*trans(results.grad);
    if(y.iresp(0)>0)
    {
        z=y.iresp(0)-1;
        results.value=beta(z)+results.value;
        if(order>0) results.grad(z)=1.0+results.grad(z);
    }
    return results;
}
