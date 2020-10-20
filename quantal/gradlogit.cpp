//Log likelihood component, gradient, and Hessian
//for graded logit model with response y and parameter beta.
#include<armadillo>
using namespace arma;
struct f2v
{
    double value;
    vec grad;
    mat hess;
};
f2v gradlogit(ivec & y,vec & beta)
{
    double d,dd;
    f2v results;
    int n,nn;
    n=beta.n_elem;
    nn=n-1;
    results.grad.set_size(n);
    results.hess.set_size(n,n);
//Check for unacceptable beta.
    if(n>1)
    {
      if(max(diff(beta))>=0.0)
      {
        results.value=datum::nan;
        results.grad.fill(datum::nan);
        results.hess.fill(datum::nan);
        return results;
      }
    }
    results.value=0.0;
    results.grad.fill(0.0);
    results.hess.fill(0.0);
    if(y(0)==n)
    {
        d=1.0/(1.0+exp(-beta(nn)));
        results.value=log(d);
        results.grad(nn)=1.0-d;
        results.hess(nn,nn)=-d*(1.0-d);
        return results;
    }
    if(y(0)==0)
    {
        d=1.0/(1.0+exp(-beta(0)));
        results.value=log(1.0-d);
        results.grad(0)=-d;
        results.hess(0,0)=-d*(1.0-d);
        return results;
    }
    dd=1.0/(1.0+exp(-beta(y(0)-1)));
    d=1.0/(1.0+exp(-beta(y(0))));
    results.value=log(dd-d);
    results.grad(y(0))=-d*(1.0-d)/(dd-d);
    results.grad(y(0)-1)=dd*(1.0-dd)/(dd-d);
    results.hess(y(0),y(0))=results.grad(y(0))*(1.0-2.0*d-results.grad(y(0)));
    results.hess(y(0)-1,y(0)-1)=results.grad(y(0)-1)*(1.0-2.0*dd-results.grad(y(0)-1));
    results.hess(y(0)-1,y(0))=-results.grad(y(0))*results.grad(y(0)-1);
    results.hess(y(0),y(0)-1)=results.hess(y(0)-1,y(0));
    return results;
}
