//Log likelihood component, gradient, and Hessian
//for graded response model with response y and parameter beta.
//If order is 0, only the function is
//found, if order is 1, then the function and gradient are found.
//If order is 2,
//then the function, gradient, and Hessian are returned.
//transform is defined as in genresp.cpp.
#include<armadillo>
using namespace arma;
struct f2v
{
    double value;
    vec grad;
    mat hess;
};
struct resp
{
    ivec iresp;
    vec dresp;
};
f2v berresp(const int & , const char & , const resp & , const vec & );
f2v gradresp(const int & order, const char & transform ,
             const resp & y, const vec & beta)
{
    double d,dd,e;
    f2v results, resultp, resultq;
    int n,nn;
    resp z, zz;
    z.iresp={0};
    zz.iresp={1};
    vec gamma(1);
    n=beta.n_elem;
    nn=n-1;
    if(order>0)
    {
        results.grad.set_size(n);
        resultp.grad.set_size(1);
        resultq.grad.set_size(1);
    }
    if(order>1)
    {
        results.hess.set_size(n,n);
        resultp.hess.set_size(1,1);
        resultq.hess.set_size(1,1);
    }
//Check for unacceptable beta.
    if(n>1)
    {
      if(max(diff(beta))>=0.0)
      {
        results.value=datum::nan;
        if(order>0) results.grad.fill(datum::nan);
        if(order>1) results.hess.fill(datum::nan);
        return results;
      }
    }
    results.value=0.0;
    if(order>0) results.grad.zeros();
    if(order>1) results.hess.zeros();
    if(y.iresp(0)==n)
    {
        gamma(0)=beta(nn);
        resultp=berresp(order, transform, zz, gamma);
        results.value=resultp.value;
        if(order>0) results.grad(nn)=resultp.grad(0);
        if(order>1) results.hess(nn,nn)=resultp.hess(0,0);
        return results;
    }
    if(y.iresp(0)==0)
    {
        gamma(0)=beta(0);
        resultp=berresp(order,transform,z,gamma);
        results.value=resultp.value;
        if(order>0) results.grad(0)=resultp.grad(0);
        if(order>1) results.hess(0,0)=resultp.hess(0,0);
        return results;
    }
    gamma(0)=beta(y.iresp(0)-1);
    resultp=berresp(order,transform,zz,gamma);
    dd=exp(resultp.value);
    gamma(0)=beta(y.iresp(0));
    resultq=berresp(order,transform,zz,gamma);
    d=exp(resultq.value);
    e=dd-d; 
    results.value=log(e);
    if(order>0)
    {
        results.grad(y.iresp(0))=-d*resultq.grad(0)/e;
        results.grad(y.iresp(0)-1)=dd*resultp.grad(0)/e;
    }
    if(order>1)
    {
        results.hess(y.iresp(0),y.iresp(0))=-d*(resultq.hess(0)
            +resultq.grad(0)*resultq.grad(0))/e
            -results.grad(y.iresp(0))*results.grad(y.iresp(0));
        results.hess(y.iresp(0)-1,y.iresp(0)-1)=dd*(resultp.hess(0)
            +resultp.grad(0)*resultp.grad(0))/e
            -results.grad(y.iresp(0)-1)*results.grad(y.iresp(0)-1);
        results.hess(y.iresp(0)-1,y.iresp(0))
            =-results.grad(y.iresp(0))*results.grad(y.iresp(0)-1);
        results.hess(y.iresp(0),y.iresp(0)-1)
            =results.hess(y.iresp(0)-1,y.iresp(0));
    }
    return results;
}
