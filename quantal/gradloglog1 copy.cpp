//Log likelihood component, gradient, and Hessian
//for graded log-log model with response y and parameter beta.
#include<armadillo>
using namespace arma;
struct f1v
{
    double value;
    vec grad;
};
f1v gradloglog1(ivec & y,vec & beta)
{
    double d,dd,p,pp;
    f1v results;
    int n,nn;
    n=beta.n_elem;
    nn=n-1;
    results.grad.set_size(n);
//Check for unacceptable beta.
    if(n>1)
    {if(max(diff(beta))>=0.0)
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
        p=-exp(-beta(nn));
        results.value=p;
        results.grad(nn)=-p;
        return results;
    }
    if(y(0)==0)
    {
        p=-exp(-beta(0));
        d=exp(p);
        results.value=log(1.0-d);
        results.grad(0)=p*d/(1.0-d);
        return results;
    }
    pp=-exp(-beta(y(0)-1));
    p=-exp(-beta(y(0)));
    d=exp(p);
    dd=exp(pp);
    results.value=log(dd-d);
    results.grad(y(0))=p*d/(dd-d);
    results.grad(y(0)-1)=-pp*dd/(dd-d);
    return results;
}
