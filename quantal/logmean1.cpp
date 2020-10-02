//Log likelihood component and its gradient
//for Poisson log-linear model with response y and parameter beta.
#include<armadillo>
using namespace arma;
struct f1v
{
    double value;
    vec grad;
};
f1v logmean1(ivec & y,vec & beta)
{
    double fy,mu;
    f1v results;
    results.grad.set_size(1);
    fy=double(y(0));
    mu=exp(beta(0));
    results.value=fy*beta(0)-mu;
    results.grad(0)=fy-mu;
    return results;
}
