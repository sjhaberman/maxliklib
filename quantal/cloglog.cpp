//Log likelihood component, gradient, and Hessian matrix
//for the complementary log-log model with response y and one-dimensional
//parameter beta.
#include<armadillo>
using namespace arma;
struct f2v
{
    double value;
    vec grad;
    mat hess;
};
f2v cloglog(ivec & y,vec & beta)
{
//Probability of response of 1.
    double p,q,r;
    f2v results;
    results.grad.set_size(1);
    results.hess.set_size(1,1);
    r=exp(beta(0));
    if(y(0)==1)
    {
        q=exp(-r);
        p=1.0-q;
        results.value=log(p);
        results.grad(0)=q*r/p;
        results.hess(0,0)=(1.0-r-results.grad(0))*results.grad(0);
    }
    else
    {
        results.value=-r;
        results.grad(0)=-r;
        results.hess(0,0)=-r;
    }
    return results;
}
