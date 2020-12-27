//Log likelihood component, gradient, and Hessian
//for cumulative model with resp1onse y and parameter beta.
//Choice of tranformation is determined by transform, with 'G' for
//log-log, 'L' for logit, and 'P' for probit.
#include<armadillo>
using namespace arma;
struct f1v
{
    double value;
    vec grad;
};
f1v berresp1(const char & , const ivec & , const vec & );
f1v cumresp1(const char & transform, const ivec & y, const vec & beta)
{
    int i,n;
    ivec z={0}, zz={1};
    vec gamma(1);
    f1v results,resultp;
    n=beta.n_elem;
    results.value=0.0;
    results.grad.set_size(n);
    results.grad.zeros();
    resultp.grad.set_size(1);
    for(i=0;i<n;i++)
    {
        gamma(0)=beta(i);
        if(i<y(0))
        {
          resultp=berresp1(transform, zz, gamma);
          results.value=results.value+resultp.value;
          results.grad(i)=resultp.grad(0);
        }
        else
        {
          resultp= berresp1(transform, z, gamma);
          results.value=results.value+resultp.value;
          results.grad(i)=resultp.grad(0);
          break;
        }
    }
    return results;
}
