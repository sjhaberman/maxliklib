//Log likelihood and its gradient and Hessian
//for response model with response y and
//predictor matrix x.
//Weight w is used.
//Model codes are found in genresp.cpp.
//If order is 0, only the function is
//found, if order is 1, then the function and gradient are found.  If order is 2,
//then the function, gradient, and Hessian are returned.   If order is 3,
//then the Hessian is replaced by the approximate Hessian
//of the Louis method.
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
struct resp
{
  ivec iresp;
  vec dresp;
};
struct xsel
{
  bool all;
  uvec list;
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
f2v genresp(const int & , const model &, const resp &, const vec &);
vec vecsel(const xsel & , const vec & );
f2v genresplik(const int & order, const std::vector<dat> & data, const vec & beta)
{
    vec beta1, lambda, u, v;
    f2v obsresults;
    f2v results;
    int i,j,jj,k,kk,order0=0,order1, p,q,r,n;
    results.value=0.0;
    p=beta.n_elem;
    n=data.size();
    if(order>0)
    {
        results.grad.set_size(p);
        results.grad.zeros();
    }
    if(order>1)
    {
        results.hess.set_size(p,p);
        results.hess.zeros();
    }
    order1=order;
    if(order>2)order1=1;
    for (i=0;i<n;i++)
    {
        r=data[i].offset.n_elem;
        q=data[i].indep.n_cols;
        lambda.set_size(r);
        lambda=data[i].offset;
        if(order>0&&q>0)obsresults.grad.set_size(q);
        if(order>1&&q>0)obsresults.hess.set_size(q,q);
        if(q>0)
        {
            beta1.set_size(q);
            beta1=vecsel(data[i].xselect,beta);
            lambda=lambda+data[i].indep*beta1;
            obsresults=genresp(order1, data[i].choice, data[i].dep, lambda);
        }
        else
        {
            obsresults=genresp(order0, data[i].choice, data[i].dep, lambda);
        }
        if(isnan(obsresults.value))
        {
            results.value=datum::nan;
            if(order>0)results.grad.fill(datum::nan);
            if(order>1)results.hess.fill(datum::nan);
            return results;
        }
        results.value=results.value+data[i].weight*obsresults.value;
        if(q>0&&order>0)
        {
            u.set_size(r);
            u=obsresults.grad;
            v.set_size(q);
            v=data[i].indep.t()*u;
            if(data[i].xselect.all)
            {
                results.grad=results.grad+data[i].weight*v;
            }
            else
            {
                results.grad.elem(data[i].xselect.list)=results.grad.elem(data[i].xselect.list)
                    +data[i].weight*v;
            }
            if(order==2)
            {
                if(data[i].xselect.all)
                {
                    results.hess=results.hess+data[i].weight*data[i].indep.t()*obsresults.hess*data[i].indep;
                }
                else
                {
                    results.hess(data[i].xselect.list,data[i].xselect.list)=
                        results.hess(data[i].xselect.list,data[i].xselect.list)
                            +data[i].weight*data[i].indep.t()*obsresults.hess*data[i].indep;
                }
            }
            if(order==3)
            {
                if(data[i].xselect.all)
                {
                    results.hess=results.hess-data[i].weight*v*v.t();
                }
                else
                {
                    results.hess(data[i].xselect.list,data[i].xselect.list)=
                        results.hess(data[i].xselect.list,data[i].xselect.list)-data[i].weight*v*v.t();
                }
            }
        }
    }
    return results;
}
