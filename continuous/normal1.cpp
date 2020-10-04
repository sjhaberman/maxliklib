//Log likelihood component and its gradient
//for noraml model with response y and parameter vector beta
//with elements beta(1) and beta(2)>0.
#include<armadillo>
using namespace arma;
struct f1v
{
    double value;
    vec grad;
};
f1v normal1(vec & y,vec & beta)
{
    double z;
    f1v results;
    results.grad.set_size(2);
    if(beta(1)<0.0)
    {
      results.value=datum::nan;
      results.grad.fill(datum::nan);
      return results;
    }
    z=beta(0)+beta(1)*y(0);
    results.value=log(beta(1))-0.5*z*z;
    results.grad(0)=-z;
    results.grad(1)=-y(0)*z+1.0/beta(1);
    return results;
}
