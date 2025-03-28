//Log likelihood component and its gradient and Hessian
//for Poisson log-linear model with response y and parameter
//vector beta.   If order is 0, only the function is
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
f2v logmean(const int & order, const resp & y, const vec & beta)
{
    double fy,mu;
    f2v results;
    if(order>0) results.grad.set_size(1);
    if(order>1) results.hess.set_size(1,1);
    fy=double(y.iresp(0));
    mu=exp(beta(0));
    results.value=fy*beta(0)-mu-lgamma(fy+1.0);
    if(order>0) results.grad(0)=fy-mu;
    if(order>1) results.hess(0,0)=-mu;
    return results;
}
