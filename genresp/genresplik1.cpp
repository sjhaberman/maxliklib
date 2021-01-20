//Log likelihood and its gradient
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
f1v genresplik1(const std::vector<dat> & data, const vec & beta)
{
    vec lambda;
    f1v obsresults;
    f1v results;
    int i,j,k,p,q,r,n;
    results.value=0.0;
    p=beta.n_elem;
    n=data.size();
    results.grad.set_size(p);
    results.grad.zeros();
    for (i=0;i<n;i++)
    {
        r=data[i].indep.n_rows;
        q=data[i].indep.n_cols;
        lambda.set_size(r);
        lambda=data[i].offset;
        obsresults.grad.set_size(q);
        if(q>0)
        {
            if(data[i].xselect.all)
            {
                lambda=lambda+data[i].indep*beta;
            }
            else
            {
                for(j=0;j<q;j++)
                {
                    k=data[i].xselect.list(j);
                    lambda=lambda+beta(k)*data[i].indep.col(j);
            }
          }
        }
        obsresults=genresp1(data[i].choice,data[i].dep,lambda);
        results.value=results.value+data[i].weight*obsresults.value;
        if(q>0)
        {
            if(data[i].xselect.all)
            {
                results.grad=results.grad
                    +data[i].weight*trans(data[i].indep)*obsresults.grad;
            }
            else
            {
                for(j=0;j<q;j++)
                {
                    k=data[i].xselect.list(j);
                    results.grad(k)=results.grad(k)
                        +data[i].weight*dot(obsresults.grad,data[i].indep.col(j));
                }
            }
        }
    }
    return results;
}
