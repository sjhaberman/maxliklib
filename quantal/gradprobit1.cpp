//Log likelihood component, gradient, and Hessian
//for graded probit model with response y and parameter beta.
#include<armadillo>
using namespace arma;
struct f1v
{
    double value;
    vec grad;
};
f1v gradprobit1(ivec & y,vec & beta)
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
        d=normcdf(beta(nn));
        results.value=log(d);
        results.grad(nn)=normpdf(beta(nn))/d;
        return results;
    }
    if(y(0)==0)
    {
        d=normcdf(beta(0));
        results.value=log(1.0-d);
        results.grad(0)=-normpdf(beta(0))/(1.0-d);
        return results;
    }
    dd=normcdf(beta(y(0)-1));
    d=normcdf(beta(y(0)));
    results.value=log(dd-d);
    results.grad(y(0))=-normpdf(beta(y(0)))/(dd-d);
    results.grad(y(0)-1)=normpdf(beta(y(0)-1))/(dd-d);
    return results;
}
