//Log likelihood component and derivative
//for complementary log-log model with response y and parameter beta.
#include<armadillo>
using namespace arma;
struct f1v
{
    double value;
    vec grad;
};
f1v cloglog1(ivec & y,vec & beta)
{
//Probability of response of 1.
    double p,q,r;
    f1v results;
    results.grad.set_size(1);
    r=exp(beta(0));
    if(y(0)==1)
    {

        q=exp(-r);
        p=1.0-q;
        results.value=log(p);
        results.grad(0)=q*r/p;
    }
    else
    {
        results.value=-r;
        results.grad(0)=-r;
    }
    return results;
}
