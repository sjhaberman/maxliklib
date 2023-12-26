//Log likelihood component, gradient, and Hessian matrix
//for the log-log model with response y and one-dimensional
//parameter beta.
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
f2v loglog(const int & order, const resp & y, const vec & beta)
{
    double p,q,r;
    f2v results;
    if(order>0) results.grad.set_size(1);
    if(order>1) results.hess.set_size(1,1);
    r=exp(-beta(0));
    if(y.iresp(0)==0)
    {
        results.value=-r;
        if(order>0) results.grad(0)=r;
        if(order>1) results.hess(0,0)=-r;
    }
    else
    {
        p=exp(-r);
        q=1.0-p;
        results.value=log(q);
        if(order>0) results.grad(0)=-r*p/q;
        if(order>1) results.hess(0,0)=-results.grad(0)*(1.0-r+results.grad(0));
    }
    return results;
}
