//Log likelihood and its gradient and Hessian
//for response model with response y and
//predictor matrix x.
//Weight w is used.
//Model codes are found in genresp.cpp.
//xselect indicates which predictors apply to which responses.
#include<armadillo>
using namespace arma;
using namespace std;
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
struct dat
{
     model choice;
     double weight;
     resp dep;
     vec offset;
     mat indep;
     xsel xselect;
};
f1v genresp1(const model &, const resp &, const vec &);
f2v genresplikl(const vector<dat> & data, const vec & beta)
{
    vec lambda,u;
    f1v obsresults;
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
              jj=data[i].xselect.list(j);
              lambda=lambda+beta(jj)*data[i].indep.col(jj);
            }
          }
        }
        obsresults=genresp1(data[i].choice,data[i].dep,lambda);
        results.value=results.value+data[i].weight*obsresults.value;
        if(data[i].xselect.all)
        {
          u.set_size(p);
          u=trans(data[i].indep)*obsresults.grad;
          results.grad=results.grad+data[i].weight*u;
          for(j=0;j<p;j++)
          {
            for(k=0;k<=j;k++)
            {
              results.hess(j,k)=results.hess(j,k)-data[i].weight*u(j)*u(k);
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
              jj=data[i].xselect.list(j);
              u(j)=dot(obsresults.grad,data[i].indep.col(jj));
              results.grad(jj)=results.grad(jj)+data[i].weight*u(j);
              for(k=0;k<=j;k++)
              {
                kk=data[i].xselect.list(k);
                results.hess(jj,kk)=results.hess(jj,kk)-data[i].weight*u(j)*u(k);
              }
            }
          }
        }
    }
    if(p>1)results.hess=symmatl(results.hess);
    return results;
}
