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
        d=1.0/(1.0+exp(-beta(beta.n_elem-1)));
        results.value=log(1.0-d);
        results.grad(beta.n_elem-1)=-d;
        return results;
    }
    if(y(0)==0)
    {
        d=1.0/(1.0+exp(-beta(0)));
        results.value=log(d);
        results.grad(0)=1.0-d;
        return results;
    }
    d=1.0/(1.0+exp(-beta(y(0)-1)));
    dd=1.0/(1.0+exp(-beta(y(0))));
    results.value=log(dd-d);
    results.grad(y(0))=dd*(1.0-dd)/(dd-d);
    results.grad(y(0)-1)=-d*(1.0-d)/(dd-d);
    return results;
}
