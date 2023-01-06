//Log likelihood component and its gradient and hessian matrix
//for logit-beta model with response y and parameter vector beta
//with positive elements beta(0) and beta(1).  If order is 0,
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
f2v logitbeta(const int & order, const resp & y, const vec & beta)
{
    double dz,tz,z,zz,zzz;
    f2v results;
    if(order>0) results.grad.set_size(2);
    if(order>1) results.hess.set_size(2,2);
    if(min(beta)<=0.0)
    {
      results.value=datum::nan;
      if(order>0) results.grad.fill(datum::nan);
      if(order>1) results.hess.fill(datum::nan);
      return results;
    }
    z=y.dresp(0);
    zz=beta(0)+beta(1);
    zzz=log(1.0+exp(z));
    results.value=beta(0)*z-zz*zzz+lgamma(zz)-lgamma(beta(0))-lgamma(beta(1));
    if(order>0)
    {
        dz=digamma(zz);
        results.grad(0)=z-zzz+dz-digamma(beta(0));
        results.grad(1)=-zzz+dz-digamma(beta(1));
    }
    if(order>1)
    {
        tz=trigamma(zz);
        results.hess(0,0)=tz-trigamma(beta(0));
        results.hess(0,1)=tz;
        results.hess(1,0)=results.hess(0,1);
        results.hess(1,1)=tz-trigamma(beta(1));
    }
    return results;
}
