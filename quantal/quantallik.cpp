//Log likelihood and its gradient and Hessian
//for quantal response model with response vector y and
//predictor matrix x.
//Weight w is used.
//Model codes are found in quantal.cpp.
//xselect indicates which predictors apply to which responses.
#include<armadillo>
using namespace arma;
struct f2v
{
    double value;
    vec grad;
    mat hess;
};
struct model
{
  char type;
  char transform;
};
struct xsel
{
  bool all;
  ivec list;
};
extern model choices[];
extern vec w;
extern ivec y[];
extern mat x[];
extern xsel xselect [];
extern vec offset[];
f2v quantal(model &,ivec &,vec &);
f2v quantallik(vec & beta)
{
    vec lambda;
    f2v obsresults;
    f2v results;
    int i,j,jj,k,kk,p,q,r,n;
    results.value=0.0;
    p=beta.n_elem;
    n=w.n_elem;
    results.grad.set_size(p);
    results.grad.zeros();
    results.hess.set_size(p,p);
    results.hess.zeros();
    for (i=0;i<n;i++)
    {
        r=offset[i].n_elem;
        lambda.set_size(r);
        lambda=offset[i];
        obsresults.grad.set_size(r);
        obsresults.hess.set_size(r,r);
        if(xselect[i].all)
        {
          lambda=lambda+x[i]*beta;
        }
        else
        {
          q=xselect[i].list.n_elem;
          if(q>0)
          {
            for(j=0;j<q;j++)
            {
              k=xselect[i].list(j);
              lambda=lambda+beta(k)*x[i].col(k);
            }
          }
        }
        obsresults=quantal(choices[i],y[i],lambda);
        results.value=results.value+w(i)*obsresults.value;
        if(xselect[i].all)
        {
          results.grad=results.grad+w(i)*trans(x[i])*obsresults.grad;
          for(j=0;j<p;j++)
          {
            for(k=0;k<=j;k++)
            {
              results.hess(j,k)=results.hess(j,k)+w(i)*dot(x[i].col(j),obsresults.hess*x[i].col(k));
            }
          }
        }
        else
        {
          if(q>0)
          {
            for(j=0;j<q;j++)
            {
              jj=xselect[i].list(j);
              results.grad(jj)=results.grad(jj)+w(i)*dot(obsresults.grad,x[i].col(jj));
              for(k=0;k<=j;k++)
              {
                kk=xselect[i].list(k);
                results.hess(jj,kk)=results.hess(jj,kk)+w(i)*dot(x[i].col(jj),obsresults.hess*x[i].col(kk));
              }
            }
          }
        }
    }
    if(p>1)results.hess=symmatl(results.hess);
    return results;
}
