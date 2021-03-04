//Log likelihood component and its gradient and hessian
//for probit model with response y and parameter beta.
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
f2v probit(const int & order, const resp & y, const vec & beta)
{
    double p,q,r;
    f2v results;
    results.grad.set_size(1);
    results.hess.set_size(1,1);
    p=normcdf(beta(0));
    q=1.0-p;
    r=normpdf(beta(0));
    if(y.iresp(0)==1)
    {
        results.value=log(p);
        if(order>0) results.grad(0)=r/p;
    }
    else
    {
        results.value=log(q);
        if(order>0) results.grad(0)=-r/q;
    }
    results.hess(0,0)=-(beta(0)+results.grad(0))*results.grad(0);
    return results;
}
