//Log likelihood component and its gradient and hessian matrix
//for beta model with response y and parameter vector beta
//with elements beta(0) and beta(1).  If order is 0,
//only the function is
//found, if order is 1, then the function and gradient are found.
//If order is 2,
//then the function, gradient, and Hessian are returned.
#include<armadillo>
#include<boost/math/special_functions/digamma.hpp>
#include<boost/math/special_functions/trigamma.hpp>
using namespace arma;
using namespace boost::math;
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
f2v betad(const int & order, const resp & y, const vec & beta)
{
    double dz,tz,y0,y1,z0,z1,zz,zzz;
    f2v results;
    if(order>0) results.grad.set_size(2);
    if(order>1) results.hess.set_size(2,2);
    y0=log(y.dresp(0));
    y1=log(1.0-y.dresp(0));
    z0=exp(beta(0));
    z1=exp(beta(1));
    zz=z0+z1;
    results.value=(z0-1.0)*y0+(z1-1.0)*y1+lgamma(zz)-lgamma(z0)-lgamma(z1);
    if(order>0)
    {
       dz=digamma(zz);
       results.grad(0)=z0*(y0+dz-digamma(z0));
       results.grad(1)=z1*(y1+dz-digamma(z1));
    }
    if(order>1)
    {
       tz=trigamma(zz);
       results.hess(0,0)=results.grad(0)+z0*z0*(tz-trigamma(z0));
       results.hess(0,1)=z0*z1*tz;
       results.hess(1,0)=results.hess(0,1);
       results.hess(1,1)=results.grad(1)+z1*z1*(tz-trigamma(z1));
    }
    return results;
}
