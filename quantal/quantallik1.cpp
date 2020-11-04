//Log likelihood and its gradient
//for quantal response model with response vector y and
//predictor matrix x.
//Weight w is used.
//Model codes are found in quantal1.cpp.
//Sparseness is not exploited.
#include<armadillo>
using namespace arma;
struct f1v
{
    double value;
    vec grad;
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
f1v quantal1(model &,ivec &,vec &);
f1v quantallik1(vec & beta)
{
    vec lambda;
    f1v obsresults;
    f1v results;
    int i,j,jj,p,q,r,n;
    results.value=0.0;
    p=beta.n_elem;
    n=w.n_elem;
    results.grad.set_size(p);
    results.grad.zeros();
    for (i=0;i<n;i++)
    {
        r=offset[i].n_elem;
        lambda.set_size(r);
        obsresults.grad.set_size(r);
        lambda=offset[i];
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
        obsresults=quantal1(choices[i],y[i],lambda);
        results.value=results.value+w(i)*obsresults.value;
        if(xselect[i].all)
        {
          results.grad=results.grad+w(i)*trans(x[i])*obsresults.grad;
        }
        else
        {
          if(q>0)
          {
            for(j=0;j<q;j++)
            {
              jj=xselect[i].list(j);
              results.grad(jj)=results.grad(jj)+w(i)*dot(obsresults.grad,x[i].col(jj));
            }
          }
        }
    }
    return results;
}
