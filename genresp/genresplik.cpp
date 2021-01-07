//Log likelihood and its gradient and Hessian
//for response model with response y and
//predictor matrix x.
//Weight w is used.
//Model codes are found in genresp.cpp.
//xselect indicates which predictors apply to which responses.
#include<armadillo>
using namespace arma;
using namespace std;
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
struct dat
{
     model choice;
     double weight;
     resp dep;
     vec offset;
     mat indep;
     xsel xselect;
};
 
f2v genresp(const model &, const resp &, const vec &);
f2v genresplik(const vector<dat> & data, const vec & beta)
{
    vec lambda;
    f2v obsresults;
    f2v results;
    int i,j,jj,k,kk,p,q,r,n;
    results.value=0.0;
    p=beta.n_elem;
    n=data.size();
    results.grad.set_size(p);
    results.grad.zeros();
    results.hess.set_size(p,p);
    results.hess.zeros();
    for (i=0;i<n;i++)
    {
        r=data[i].offset.n_elem;
        lambda.set_size(r);
        lambda=data[i].offset;
        obsresults.grad.set_size(r);
        obsresults.hess.set_size(r,r);
        if(data[i].xselect.all)
        {
          lambda=lambda+data[i].indep*beta;
        }
        else
        {
          q=data[i].xselect.list.n_elem;
          if(q>0)
          {
            for(j=0;j<q;j++)
            {
              k=data[i].xselect.list(j);
              lambda=lambda+beta(k)*data[i].indep.col(k);
            }
          }
        }
        obsresults=genresp(data[i].choice,data[i].dep,lambda);
        results.value=results.value+data[i].weight*obsresults.value;
        if(data[i].xselect.all)
        {
          results.grad=results.grad
             +data[i].weight*trans(data[i].indep)*obsresults.grad;
          for(j=0;j<p;j++)
          {
            for(k=0;k<=j;k++)
            {
              results.hess(j,k)=results.hess(j,k)
                 +data[i].weight*dot(data[i].indep.col(j),obsresults.hess*data[i].indep.col(k));
            }
          }
        }
        else
        {
          if(q>0)
          {
            for(j=0;j<q;j++)
            {
              jj=data[i].xselect.list(j);
              results.grad(jj)=results.grad(jj)
                 +data[i].weight*dot(obsresults.grad,data[i].indep.col(jj));
              for(k=0;k<=j;k++)
              {
                kk=data[i].xselect.list(k);
                results.hess(jj,kk)=results.hess(jj,kk)
                   +data[i].weight*
                   dot(data[i].indep.col(jj),obsresults.hess*data[i].indep.col(kk));
              }
            }
          }
        }
    }
    if(p>1)results.hess=symmatl(results.hess);
    return results;
}
