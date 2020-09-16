//Log likelihood component, gradient, and Hessian matrix
//for complementary log-log model with response y and one-dimensional
//parameter beta.
#include<armadillo>
using namespace arma;
struct f2v
{
    double value;
    vec grad;
    mat hess;
};
f2v cloglog(int y,vec beta)
{
//Probability of response of 1.
    double p,q,r;
    f2v results;
    results.grad.set_size(1);
    results.hess.set_size(1,1);
    q=exp(-exp(beta(0)));
    p=1.0-q;
    if(y==1)
    {
        r=exp(beta(0))/p;
        results.value=log(p);
        results.grad(0)=r*q;
        results.hess(0,0)=r*q*(1.0-r);
    }
    else
    {
        results.value=log(q);
        results.grad(0)=results.value;
        results.hess(0,0)=results.value;
    }
    return results;
}
