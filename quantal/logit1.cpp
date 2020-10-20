//Log likelihood component and gradient
//for logit model with response y and one-dimensional parameter beta.
#include<armadillo>
using namespace arma;
struct f1v
{
    double value;
    vec grad;
};
f1v logit1(ivec & y,vec & beta)
{
//Probability of response of 1.
    double p,q;
    f1v results;
    results.grad.set_size(1);
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
    return results;
}
