//Log likelihood component, gradient, and Hessian
//for graded probit model with response y and parameter beta.
#include<armadillo>
using namespace arma;
struct f2v
{
    double value;
    vec grad;
    mat hess;
};
f2v gradprobit(ivec & y,vec & beta)
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
        d=normcdf(beta(nn));
        results.value=log(d);
        results.grad(nn)=normpdf(beta(nn))/d;
        results.hess(nn,nn)=(-beta(nn)-results.grad(nn))*results.grad(nn);
        return results;
    }
    if(y(0)==0)
    {
        d=normcdf(beta(0));
        results.value=log(1.0-d);
        results.grad(0)=-normpdf(beta(0))/(1.0-d);
        results.hess(0,0)=(-beta(0)-results.grad(0))*results.grad(0);
        return results;
    }
    dd=normcdf(beta(y(0)-1));
    d=normcdf(beta(y(0)));
    results.value=log(dd-d);
    results.grad(y(0))=-normpdf(beta(y(0)))/(dd-d);
    results.grad(y(0)-1)=normpdf(beta(y(0)-1))/(dd-d);
    results.hess(y(0),y(0))=(beta(y(0))-results.grad(y(0)))*results.grad(y(0));
    results.hess(y(0)-1,y(0)-1)=(-beta(y(0)-1)-results.grad(y(0)-1))*results.grad(y(0)-1);
    results.hess(y(0)-1,y(0))=-results.grad(y(0))*results.grad(y(0)-1);
    results.hess(y(0),y(0)-1)=results.hess(y(0)-1,y(0));
    return results;
}
