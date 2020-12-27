//Log likelihood component, gradient, and Hessian
//for cumulative model with response y and parameter beta.
//Choice of tranformation is determined by transform, with 'G' for
//log-log, 'L' for logit, and 'P' for probit.
#include<armadillo>
using namespace arma;
struct f2v
{
    double value;
    vec grad;
    mat hess;
};
f2v berresp(const char & , const ivec & , const vec & );
f2v cumresp(const char & transform, const ivec & y, const vec & beta)
{
    int i,n;
    ivec z={0}, zz={1};
    vec gamma(1);
    f2v results,resultp;
    n=beta.n_elem;
    results.value=0.0;
    results.grad.set_size(n);
    results.grad.zeros();
    results.hess.set_size(n,n);
    results.hess.zeros();
    resultp.grad.set_size(1);
    resultp.hess.set_size(1,1);
    for(i=0;i<n;i++)
    {
        gamma(0)=beta(i);
        if(i<y(0))
        {
          resultp=berresp(transform, zz, gamma);
          results.value=results.value+resultp.value;
          results.grad(i)=resultp.grad(0);
          results.hess(i)=resultp.hess(0,0);
        }
        else
        {
          resultp= berresp(transform, z, gamma);
          results.value=results.value+resultp.grad(0);
          results.grad(i)=resultp.grad(0);
          results.hess(i,i)=resultp.hess(0,0);
          break;
        }
    }
    return results;
}
