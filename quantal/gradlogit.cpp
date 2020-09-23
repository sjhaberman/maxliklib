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
    results.grad.set_size(beta.n_elem);
    results.hess.set_size(beta.n_elem,beta.n_elem);
//Check for unacceptable beta.
    if(min(diff(beta))<=0.0)
    {
        results.value=datum::nan;
        results.grad.fill(datum::nan);
        results.hess.fill(datum::nan);
        return results;
    }
    results.value=0.0;
    results.grad.fill(0.0);
    results.hess.fill(0.0);
    if(y(0)==beta.n_elem)
    {
        d=1.0/(1.0+exp(-beta(beta.n_elem-1)));
        results.value=log(1.0-d);
        results.grad(beta.n_elem-1)=-d;
        results.hess(beta.n_elem-1,beta.n_elem-1)=-d*(1.0-d);
        return results;
    }
    if(y(0)==0)
    {
        d=1.0/(1.0+exp(-beta(0)));
        results.value=log(d);
        results.grad(0)=1.0-d;
        results.hess(0,0)=-d*(1.0-d);
        return results;
    }
    d=1.0/(1.0+exp(-beta(y(0)-1)));
    dd=1.0/(1.0+exp(-beta(y(0))));
    results.value=log(dd-d);
    results.grad(y(0))=dd*(1.0-dd)/(dd-d);
    results.grad(y(0)-1)=-d*(1.0-d)/(dd-d);
    results.hess(y(0),y(0))=dd*(1.0-dd)*(1.0-2.0*dd)/(dd-d)-results.grad(y(0))*results.grad(y(0));
    results.hess(y(0)-1,y(0)-1)=-d*(1.0-d)*(1.0-2.0*d)/(dd-d)-results.grad(y(0)-1)*results.grad(y(0)-1);
    results.hess(y(0)-1,y(0))=-results.grad(y(0))*results.grad(y(0)-1);
    results.hess(y(0),y(0)-1)=results.hess(y(0)-1,y(0));
    return results;
}
