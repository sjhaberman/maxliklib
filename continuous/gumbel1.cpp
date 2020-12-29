//Log likelihood component and its gradient x
//for gumbel model with response y and parameter vector beta
//with elements beta(1) and beta(2)>0.
#include<armadillo>
using namespace arma;
struct f1v
{
    double value;
    vec grad;
};
f1v gumbel1(const vec & y, const vec & beta)
{
    double z,zz;
    f1v results;
    results.grad.set_size(2);
    if(beta(1)<0.0)
    {
      results.value=datum::nan;
      results.grad.fill(datum::nan);
      return results;
    }
    z=beta(0)+beta(1)*y(0);
    zz=exp(-z);
    results.value=log(beta(1))-z-zz;
    results.grad(0)=zz-1.0;
    results.grad(1)=y(0)*results.grad(0)+1.0/beta(1);
    return results;
}
