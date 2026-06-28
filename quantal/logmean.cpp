//Log likelihood component and its gradient and Hessian
//for Poisson log-linear model with response y and parameter
//vector beta.   If order is 0, only the function is
//found, if order is 1, then the function and gradient are found.
//If order is 2,
//then the function, gradient, and Hessian are returned.
#include<armadillo>
using namespace arma;
struct f2v{double value; vec grad; mat hess;};
f2v logmean(const int & order, const vec & y, const vec & beta){
    double fy,mu;
    f2v results;
    fy=y(0);
    mu=exp(beta(0));
    results.value=fy*beta(0)-mu-lgamma(fy+1.0);
    if(order==0) return results;
    results.grad.set_size(1);
    results.grad(0)=fy-mu;
    if(order==1) return results;
    results.hess.set_size(1,1);
    results.hess(0,0)=-mu;
    return results;
}
