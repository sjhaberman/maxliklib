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
f2v berresp(const char & , const ivec & , const vec & );
f2v gradresp(const char & transform , const ivec & y, const vec & beta)
{
    double d,dd,e;
    f2v results, resultp,resultq;
    int n,nn;
    ivec z={0}, zz={1};
    vec gamma(1);
    n=beta.n_elem;
    nn=n-1;
    results.grad.set_size(n);
    results.hess.set_size(n,n);
    resultp.grad.set_size(1);
    resultp.hess.set_size(1,1);
    resultq.grad.set_size(1);
    resultq.hess.set_size(1,1);
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
    results.grad.zeros();
    results.hess.zeros();
    if(y(0)==n)
    {
        gamma(0)=beta(nn);
        resultp=berresp(transform,zz,gamma);
        results.value=resultp.value;
        results.grad(nn)=resultp.grad(0);
        results.hess(nn,nn)=resultp.hess(0,0);
        return results;
    }
    if(y(0)==0)
    {
        gamma(0)=beta(0);
        resultp=berresp(transform,z,gamma);
        results.value=resultp.value;
        results.grad(0)=resultp.grad(0);
        results.hess(0,0)=resultp.hess(0,0);
        return results;
    }
    gamma(0)=beta(y(0)-1);
    resultp=berresp(transform,zz,gamma);
    dd=exp(resultp.value);
    gamma(0)=beta(y(0));
    resultq=berresp(transform,zz,gamma);
    d=exp(resultq.value);
    e=dd-d; 
    results.value=log(e);
    results.grad(y(0))=-d*resultq.grad(0)/e;
    results.grad(y(0)-1)=dd*resultp.grad(0)/e;
    results.hess(y(0),y(0))=-d*(resultq.hess(0)+d*resultq.grad(0))
       -results.grad(y(0))*results.grad(y(0));
    results.hess(y(0)-1,y(0)-1)=-dd*(resultp.hess(0)+d*resultp.grad(0))
       -results.grad(y(0))*results.grad(y(0));
    results.hess(y(0)-1,y(0))=-results.grad(y(0))*results.grad(y(0)-1);
    results.hess(y(0),y(0)-1)=results.hess(y(0)-1,y(0));
    return results;
}
