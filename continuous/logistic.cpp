//Log likelihood component and its gradient and hessian matrix
//for logistic model with response y and parameter vector beta
//with elements beta(1) and beta(2)>0.
//If order is 0, only the function is
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
f2v logistic(const int & order, const resp & y, const vec & beta)
{
    double z,zz;
    f2v results;
    if(order>0) results.grad.set_size(2);
    if(order>1) results.hess.set_size(2,2);
    if(beta(1)<0.0)
    {
      results.value=datum::nan;
      if(order>0) results.grad.fill(datum::nan);
      if(order>1) results.hess.fill(datum::nan);
      return results;
    }
    z=beta(0)+beta(1)*y.dresp(0);
    zz=1.0/(1.0+exp(-z));
    results.value=-z+2.0*log(zz)+log(beta(1));
    if(order>0)
    {
        results.grad(0)=1.0-2.0*zz;
        results.grad(1)=y.dresp(0)*results.grad(0)+1.0/beta(1);
    }
    if(order>1)
    {
        results.hess(0,0)=-2.0*zz*(1.0-zz);
        results.hess(0,1)=y.dresp(0)*results.hess(0,0);
        results.hess(1,0)=results.hess(0,1);
        results.hess(1,1)=y.dresp(0)*results.hess(0,1)-1.0/(beta(1)*beta(1));
    }
    return results;
}
