//Log likelihood component and derivative
//for complementary log-log model with response y and parameter beta.
#include<armadillo>
using namespace arma;
struct f1v
{
    double value;
    vec grad;
};
f1v cloglog1(int y,vec beta)
{
//Probability of response of 1.
    double p,q,r;
    f1v results;
    results.grad.set_size(1);
    q=exp(-exp(beta(0)));
    p=1.0-q;
    
    if(y==1)
    {
        r=exp(beta(0))/p;
        results.value=log(p);
        results.grad(0)=r*q;
    }
    else
    {
        results.value=log(q);
        results.grad(0)=results.value;
    }
    return results;
}
