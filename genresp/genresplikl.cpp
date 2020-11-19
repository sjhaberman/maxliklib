//Log likelihood and its gradient and Hessian
//for response model with response y and
//predictor matrix x.
//Weight w is used.
//Model codes are found in genresp.cpp.
//xselect indicates which predictors apply to which responses.
#include<armadillo>
using namespace arma;
struct f1v
{
    double value;
    vec grad;
};
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
struct resp
{
  ivec iresp;
  vec dresp;
};
struct xsel
{
  bool all;
  ivec list;
};
extern model choices[];
extern vec w;
extern resp y[];
extern mat x[];
extern xsel xselect [];
extern vec offset[];
f1v genresp1(model &,resp &,vec &);
f2v genresplikl(vec & beta)
{
    vec lambda,u;
    f1v obsresults;
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
              jj=xselect[i].list(j);
              lambda=lambda+beta(jj)*x[i].col(jj);
            }
          }
        }
        obsresults=genresp1(choices[i],y[i],lambda);
        results.value=results.value+w(i)*obsresults.value;
        if(xselect[i].all)
        {
          u.set_size(p);
          u=trans(x[i])*obsresults.grad;
          results.grad=results.grad+w(i)*u;
          for(j=0;j<p;j++)
          {
            for(k=0;k<=j;k++)
            {
              results.hess(j,k)=results.hess(j,k)-w(i)*u(j)*u(k);
            }
          }
        }
        else
        {
          if(q>0)
          {
            for(j=0;j<q;j++)
            {
              u.set_size(q);
              jj=xselect[i].list(j);
              u(j)=dot(obsresults.grad,x[i].col(jj));
              results.grad(jj)=results.grad(jj)+w(i)*u(j);
              for(k=0;k<=j;k++)
              {
                kk=xselect[i].list(k);
                results.hess(jj,kk)=results.hess(jj,kk)-w(i)*u(j)*u(k);
              }
            }
          }
        }
    }
    if(p>1)results.hess=symmatl(results.hess);
    return results;
}
