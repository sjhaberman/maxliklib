//Log likelihood component and its gradient and Hessian
//for Poisson log-linear model with response y and parameter
//vector beta.
#include<armadillo>
using namespace arma;
struct f2v
{
    double value;
    vec grad;
    mat hess;
};
f2v logmean(ivec & y,vec & beta)
{
    double fy,mu;
    f2v results;
    results.grad.set_size(1);
    results.hess.set_size(1,1);
    fy=double(y(0));
    mu=exp(beta(0));
    results.value=fy*beta(0)-mu;
    results.grad(0)=fy-mu;
    results.hess(0,0)=-mu;
    return results;
}
