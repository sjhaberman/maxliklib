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
        d=normcdf(beta(beta.n_elem-1));
        results.value=log(1.0-d);
        results.grad(beta.n_elem-1)=-normpdf(beta(beta.n_elem-1))/(1.0-d);
        results.hess(beta.n_elem-1,beta.n_elem-1)=(-beta(beta.n_elem-1)-results.grad(beta.n_elem-1))*results.grad(beta.n_elem-1);
        return results;
    }
    if(y(0)==0)
    {
        d=normcdf(beta(0));
        results.value=log(d);
        results.grad(0)=normpdf(beta(0))/d;
        results.hess(0,0)=(-beta(0)-results.grad(0))*results.grad(0);
        return results;
    }
    d=normcdf(beta(y(0)-1));
    dd=normcdf(beta(y(0)));
    results.value=log(dd-d);
    results.grad(y(0))=normpdf(beta(y(0)))/(dd-d);
    results.grad(y(0)-1)=-normpdf(beta(y(0)-1))/(dd-d);
    results.hess(y(0),y(0))=(-beta(y(0))-results.grad(y(0)))*results.grad(y(0));
    results.hess(y(0)-1,y(0)-1)=(beta(y(0)-1)-results.grad(y(0)-1))*results.grad(y(0)-1);
    results.hess(y(0)-1,y(0))=-results.grad(y(0))*results.grad(y(0)-1);
    results.hess(y(0),y(0)-1)=results.hess(y(0)-1,y(0));
    return results;
}
