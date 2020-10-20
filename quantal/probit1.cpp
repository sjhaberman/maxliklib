//Log likelihood component and its gradient
//for probit model with response y and parameter beta.
#include<armadillo>
using namespace arma;
struct f1v
{
    double value;
    vec grad;
};
f1v probit1(ivec & y,vec & beta)
{
    double p,q,r;
    f1v results;
    results.grad.set_size(1);
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
    return results;
}
