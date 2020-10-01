//Log likelihood component, gradient, and Hessian
//for graded complementary log-log model with response y and parameter beta.
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
        p=exp(beta(beta.n_elem-1));
        results.value=-p;
        results.grad(beta.n_elem-1)=-p;
        results.hess(beta.n_elem-1,beta.n_elem-1)=-p;
        return results;
    }
    if(y(0)==0)
    {
        p=exp(beta(beta.n_elem-1));
        d=1.0-exp(-p);
        results.value=log(d);
        results.grad(0)=p*(1.0-d)/d;
        results.hess(0,0)=results.grad(0)*(1.0-p/d);
        return results;
    }
    p=exp(beta(y(0)-1));
    pp=exp(beta(y(0)));
    d=exp(-p);
    dd=exp(-pp);
    results.value=log(d-dd);
    results.grad(y(0))=pp*dd/(d-dd);
    results.grad(y(0)-1)=-p*d/(d-dd);
    results.hess(y(0),y(0))=results.grad(y(0))*(1.0-pp-results.grad(y(0)));
    results.hess(y(0)-1,y(0)-1)=results.grad(y(0)-1)*(1.0-p-results.grad(y(0)-1));
    results.hess(y(0)-1,y(0))=-results.grad(y(0))*results.grad(y(0)-1);
    results.hess(y(0),y(0)-1)=results.hess(y(0)-1,y(0));
    return results;
}
