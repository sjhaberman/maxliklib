//Log likelihood and its gradient
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
f1v genresplik1(const vector<dat> & data, const vec & beta)
{
    vec lambda;
    f1v obsresults;
    f1v results;
    int i,j,jj,p,q,r,n;
    results.value=0.0;
    p=beta.n_elem;
    n=data.size();
    results.grad.set_size(p);
    results.grad.zeros();
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
          results.grad=results.grad
             +data[i].weight*trans(data[i].indep)*obsresults.grad;
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
            }
          }
        }
    }
    return results;
}
