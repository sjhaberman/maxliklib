//Log likelihood and its gradient
//for response model with response y and
//predictor matrix x.
//Weight w is used.
//Model codes are found in genresp1.cpp.
//xselect indicates which predictors apply to which responses.
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
f1v genresplik1(vec & beta)
{
    vec lambda;
    f1v obsresults;
    f1v results;
    int i,j,k,p,q,r,n;
    results.value=0.0;
    p=beta.n_elem;
    n=w.n_elem;
    results.grad.set_size(p);
    results.grad.zeros();
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
              k=xselect[i].list(j);
              lambda=lambda+beta(k)*x[i].col(k);
            }
          }
        }
        obsresults=genresp1(choices[i],y[i],lambda);
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
              k=xselect[i].list(j);
              results.grad(k)=results.grad(k)+w(i)*dot(obsresults.grad,x[i].col(k));
            }
          }
        }
    }
    return results;
}
