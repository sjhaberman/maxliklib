//Log likelihood component and its gradient and hessian
//for probit model with response y and parameter beta.
#include<armadillo>
using namespace arma;
struct f2v
{
    double value;
    vec grad;
    mat hess;
};
f2v probit(ivec & y,vec & beta)
{
    double p,q,r;
    f2v results;
    results.grad.set_size(1);
    results.hess.set_size(1,1);
    p=normcdf(beta(0));
    q=1.0-p;
    r=normpdf(beta(0));
    if(y(0)==1)
    {
        results.value=log(p);
        results.grad(0)=r/p;
    }
    else
    {
        results.value=log(q);
        results.grad(0)=-r/q;
    }
    results.hess(0,0)=-(beta(0)+results.grad(0))*results.grad(0);
    return results;
}
