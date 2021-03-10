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
f2v genresp(const int & , const model &, const resp &, const vec &);
f2v genresplik(const int & order, const std::vector<dat> & data, const vec & beta)
{
    vec lambda, u;
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
            if(data[i].xselect.all&&p==q)
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
            obsresults=genresp(order1, data[i].choice, data[i].dep, lambda);
        }
        else
        {
            obsresults=genresp(order0, data[i].choice, data[i].dep, lambda);
        }
        results.value=results.value+data[i].weight*obsresults.value;
        if(q>0&&order>0)
        {
            u.set_size(q);
            if(data[i].xselect.all&&p==q)
            {
                u=data[i].indep.t()*obsresults.grad;
                results.grad=results.grad+data[i].weight*u;
                if(order==2)
                {
                    for(j=0;j<p;j++)
                    {
                        for(k=0;k<=j;k++)
                        {
                            results.hess(j,k)=results.hess(j,k)
                                +data[i].weight*dot(data[i].indep.col(j),obsresults.hess*data[i].indep.col(k));
                            if(j!=k)results.hess(k,j)=results.hess(j,k);
                        }
                    }
                }
                if(order==3)
                {
                    for(j=0;j<p;j++)
                    {
                        for(k=0;k<=j;k++)
                        {
                            results.hess(j,k)=results.hess(j,k)-data[i].weight*u(j)*u(k);
                            if(j!=k)results.hess(k,j)=results.hess(j,k);
                        }
                    }
                }
            }
            else
            {
                for(j=0;j<q;j++)
                {
                    jj=data[i].xselect.list(j);
                    u(j)=dot(obsresults.grad,data[i].indep.col(jj));
                    results.grad(jj)=results.grad(jj)+data[i].weight*u(j);
                    if(order==2)
                    {
                        for(k=0;k<=j;k++)
                        {
                            kk=data[i].xselect.list(k);
                            results.hess(jj,kk)=results.hess(jj,kk)
                                +data[i].weight*
                                dot(data[i].indep.col(j),obsresults.hess*data[i].indep.col(k));
                            if(jj!=kk)results.hess(kk,jj)=results.hess(jj,kk);
                        }
                    }
                    if(order==3)
                    {
                        for(k=0;k<=j;k++)
                        {
                            kk=data[i].xselect.list(k);
                            results.hess(jj,kk)=results.hess(jj,kk)-data[i].weight*u(j)*u(k);
                            if(jj!=kk)results.hess(kk,jj)=results.hess(jj,kk);
                        }
                    }
                }
            }
        }
    }
    return results;
}
