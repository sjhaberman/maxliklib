//Log likelihood component and its gradient and hessian matrix
//for noraml model with response y and parameter vector beta
//with elements beta(1) and beta(2)>0.
#include<armadillo>
using namespace arma;
struct f2v
{
    double value;
    vec grad;
    mat hess;
};
f2v normal(vec & y,vec & beta)
{
    double z,zz;
    f2v results;
    results.grad.set_size(2);
    results.hess.set_size(2,2);
    if(beta(1)<0.0)
    {
      results.value=datum::nan;
      results.grad.fill(datum::nan);
      results.hess.fill(datum::nan);
      return results;
    }
    z=beta(0)+beta(1)*y(0);
    results.value=log(beta(1))-0.5*z*z;
    results.grad(0)=-z;
    results.grad(1)=-y(0)*z+1.0/beta(1);
    results.hess(0,0)=-1.0;
    results.hess(0,1)=-y(0);
    results.hess(1,0)=results.hess(0,1);
    results.hess(1,1)=y(0)*results.hess(0,1)-1.0/(beta(1)*beta(1));
    return results;
}
