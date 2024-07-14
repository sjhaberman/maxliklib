//Log likelihood component, gradient, and Hessian
//for cumulative model with response y and parameter beta.
//Choice of tranformation is determined by transform, with 'G' for
//complementary log-log, 'H' for log-log,
//'L' for logit, and 'N' for probit.  If order is 0, only the function is
//found, if order is 1, then the function and gradient are found.  If order is 2,
//then the function, gradient, and Hessian are returned.
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
f2v cumresp(const int & order, const char & transform, const resp & y,
            const vec & beta)
{
    int i,n;
    resp z, zz;
    z.iresp={0};
    zz.iresp={1};
    vec gamma(1);
    f2v results, resultp;
    n=beta.n_elem;
    results.value=0.0;
    if(order>0)
    {
        results.grad.set_size(n);
        results.grad.zeros();
        resultp.grad.set_size(1);
    }
    if(order>1)
    {
        results.hess.set_size(n,n);
        results.hess.zeros();
        resultp.hess.set_size(1,1);
    }
    for(i=0;i<n;i++)
    {
        gamma(0)=beta(i);
        if(i<y.iresp(0))
        {
          resultp=berresp(order, transform, zz, gamma);
          results.value=results.value+resultp.value;
          if(order>0) results.grad(i)=resultp.grad(0);
          if(order>1) results.hess(i,i)=resultp.hess(0,0);
        }
        else
        {
          resultp= berresp(order, transform, z, gamma);
          results.value=results.value+resultp.value;
          if(order>0) results.grad(i)=resultp.grad(0);
          if(order>1) results.hess(i,i)=resultp.hess(0,0);
          break;
        }
    }
    return results;
}
