//Log likelihood component, gradient, and Hessian
//for graded logit model with response y and parameter beta.
#include<armadillo>
using namespace arma;
struct f1v
{
    double value;
    vec grad;
};
f1v gradlogit1(ivec & y,vec & beta)
{
    double d,dd;
    f1v results;
    int n,nn;
    n=beta.n_elem;
    nn=n-1;
    results.grad.set_size(n);
//Check for unacceptable beta.
    if(n>1)
    {
      if(max(diff(beta))>=0.0)
      {
        results.value=datum::nan;
        results.grad.fill(datum::nan);
        return results;
      }
    }
    results.value=0.0;
    results.grad.zeros();
    if(y(0)==n)
    {
        d=1.0/(1.0+exp(-beta(nn)));
        results.value=log(d);
        results.grad(nn)=1.0-d;
        return results;
    }
    if(y(0)==0)
    {
        d=1.0/(1.0+exp(-beta(0)));
        results.value=log(1.0-d);
        results.grad(0)=-d;
        return results;
    }
    dd=1.0/(1.0+exp(-beta(y(0)-1)));
    d=1.0/(1.0+exp(-beta(y(0))));
    results.value=log(dd-d);
    results.grad(y(0))=-d*(1.0-d)/(dd-d);
    results.grad(y(0)-1)=dd*(1.0-dd)/(dd-d);
    return results;
}
