//Log likelihood component, gradient, and Hessian
//for graded probit model with response y and parameter beta.
#include<armadillo>
using namespace arma;
struct f1v
{
    double value;
    vec grad;
};
f1v berresp1(const char & , const ivec & , const vec & );
f1v gradresp1(const char & transform , const ivec & y, const vec & beta)
{
    double d,dd,e;
    f1v results, resultp,resultq;
    int n,nn;
    ivec z={0}, zz={1};
    vec gamma(1);
    n=beta.n_elem;
    nn=n-1;
    results.grad.set_size(n);
    resultp.grad.set_size(1);
    resultq.grad.set_size(1);
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
        gamma(0)=beta(nn);
        resultp=berresp1(transform,zz,gamma);
        results.value=resultp.value;
        results.grad(nn)=resultp.grad(0);
        return results;
    }
    if(y(0)==0)
    {
        gamma(0)=beta(0);
        resultp=berresp1(transform,z,gamma);
        results.value=resultp.value;
        results.grad(0)=resultp.grad(0);
        return results;
    }
    gamma(0)=beta(y(0)-1);
    resultp=berresp1(transform,zz,gamma);
    dd=exp(resultp.value);
    gamma(0)=beta(y(0));
    resultq=berresp1(transform,zz,gamma);
    d=exp(resultq.value);
    e=dd-d; 
    results.value=log(e);
    results.grad(y(0))=-d*resultq.grad(0)/e;
    results.grad(y(0)-1)=dd*resultp.grad(0)/e;
    return results;
}
