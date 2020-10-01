//Log likelihood component, gradient, and Hessian matrix
//for logit model with response y and one-dimensional parameter beta.
#include<armadillo>
using namespace arma;
struct f2v
{
    double value;
    vec grad;
    mat hess;
};
f2v logit(ivec & y,vec & beta)
{
//Probability of response of 1.
    double p,q;
    f2v results;
    results.grad.set_size(1);
    results.hess.set_size(1,1);
    p=1.0/(1.0+exp(-beta(0)));
    q=1.0-p;
    if(y(0)==1)
    {
        results.value=log(p);
    }
    else
    {
        results.value=log(q);
    }
    results.grad(0)=double(y(0))-p;
    results.hess(0,0)=-p*q;
    return results;
}
