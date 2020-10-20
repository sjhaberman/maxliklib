//Log likelihood component, gradient, and Hessian
//for graded log-log model with response y and parameter beta.
#include<armadillo>
using namespace arma;
struct f2v
{
    double value;
    vec grad;
    mat hess;
};
f2v gradloglog(ivec & y,vec & beta)
{
    double d,dd,p,pp;
    int n,nn;
    n=beta.n_elem;
    nn=n-1;
    f2v results;
    results.grad.set_size(n);
    results.hess.set_size(n,n);
//Check for unacceptable beta.
    if(n>1)
    {if(max(diff(beta))>=0.0)
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
        p=-exp(-beta(nn));
        results.value=p;
        results.grad(nn)=-p;
        results.hess(nn,nn)=p;
        return results;
    }
    if(y(0)==0)
    {
        p=-exp(-beta(0));
        d=exp(p);
        results.value=log(1.0-d);
        results.grad(0)=p*d/(1.0-d);
        results.hess(0,0)=-results.grad(0)*(1.0+p+results.grad(0));
        return results;
    }
    pp=-exp(-beta(y(0)-1));
    p=-exp(-beta(y(0)));
    d=exp(p);
    dd=exp(pp);
    results.value=log(dd-d);
    results.grad(y(0))=p*d/(dd-d);
    results.grad(y(0)-1)=-pp*dd/(dd-d);
    results.hess(y(0),y(0))=-results.grad(y(0))*(1.0+p+results.grad(y(0)));
    results.hess(y(0)-1,y(0)-1)=-results.grad(y(0)-1)*(1.0+pp+results.grad(y(0)-1));
    results.hess(y(0)-1,y(0))=-results.grad(y(0))*results.grad(y(0)-1);
    results.hess(y(0),y(0)-1)=results.hess(y(0)-1,y(0));
    return results;
}
