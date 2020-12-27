//Log likelihood component, gradient, and Hessian matrix
//for the log-log model with response y and one-dimensional
//parameter beta.
#include<armadillo>
using namespace arma;
struct f1v
{
    double value;
    vec grad;
};
f1v loglog1(const ivec & y, const vec & beta)
{
//Probability of response of 1.
    double p,q,r;
    f1v results;
    results.grad.set_size(1);
    r=exp(-beta(0));
    if(y(0)==1)
    {
        results.value=-r;
        results.grad(0)=r;
    }
    else
    {
        p=exp(-r);
        q=1.0-p;
        results.value=log(q);
        results.grad(0)=-r*p/q;
    }
    return results;
}
