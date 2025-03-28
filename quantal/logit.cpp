//Log likelihood component, gradient, and Hessian matrix
//for logit model with response y and one-dimensional parameter beta.
// If order is 0, only the function is
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
f2v logit(const int & order, const resp & y, const vec & beta)
{
//Probability of response of 1.
    double p,q;
    f2v results;
    if(order>0) results.grad.set_size(1);
    if(order>1) results.hess.set_size(1,1);
    p=1.0/(1.0+exp(-beta(0)));
    q=1.0-p;
    if(y.iresp(0)==1)
    {
        results.value=log(p);
    }
    else
    {
        results.value=log(q);
    }
    if(order>0) results.grad(0)=double(y.iresp(0))-p;
    if(order>1) results.hess(0,0)=-p*q;
    return results;
}
