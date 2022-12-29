//Log likelihood component and its gradient and hessian matrix
//for log-gamma model with response y and parameter vector beta
//with positive elements beta(1) and beta(2).  If order is 0,
//only the function is
//found, if order is 1, then the function and gradient are found.
//If order is 2,
//then the function, gradient, and Hessian are returned.
#include<armadillo>
#include<boost/math/special_functions/digamma.hpp>
#include<boost/math/special_functions/trigamma.hpp>
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
f2v loggamma(const int & order, const resp & y, const vec & beta)
{
    double z,zz;
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
    zz=exp(z);
    results.value=beta(0)*z-beta(1)*zz+beta(0)*log(beta(1))-lgamma(beta(0));
    if(order>0)
    {
        results.grad(0)=z+log(beta(1))-boost::math::digamma(beta(0));
        results.grad(1)=-zz+beta(0)/beta(1);
    }
    if(order>1)
    {
        results.hess(0,0)=-boost::math::trigamma(beta(0));
        results.hess(0,1)=1.0/beta(1);
        results.hess(1,0)=results.hess(0,1);
        results.hess(1,1)=-beta(0)/(beta(1)*beta(1));
    }
    return results;
}
