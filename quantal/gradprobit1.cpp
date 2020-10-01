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
    results.grad.set_size(beta.n_elem);
//Check for unacceptable beta.
    if(min(diff(beta))<=0.0)
    {
        results.value=datum::nan;
        results.grad.fill(datum::nan);
        return results;
    }
    results.value=0.0;
    results.grad.fill(0.0);
    if(y(0)==beta.n_elem)
    {
        d=normcdf(beta(beta.n_elem-1));
        results.value=log(1.0-d);
        results.grad(beta.n_elem-1)=-normpdf(beta(beta.n_elem-1))/(1.0-d);
        return results;
    }
    if(y(0)==0)
    {
        d=normcdf(beta(0));
        results.value=log(d);
        results.grad(0)=normpdf(beta(0))/d;
        return results;
    }
    d=normcdf(beta(y(0)-1));
    dd=normcdf(beta(y(0)));
    results.value=log(dd-d);
    results.grad(y(0))=normpdf(beta(y(0)))/(dd-d);
    results.grad(y(0)-1)=-normpdf(beta(y(0)-1))/(dd-d);
    return results;
}
