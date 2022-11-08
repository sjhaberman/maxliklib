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
//obssel indicates which observations to consider.
#include<armadillo>
using namespace std;
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
void addsel(const int & , const xsel & , const f2v & , f2v & , const double & );
f2v linsel(const int & , const f2v & , const mat & );
vec vecsel(const xsel & , const vec & );
f2v genresplik(const int & order, const vector<dat> & data, const xsel & obssel,
    const vec & beta)
{
    vec beta1, lambda, u, v;
    f2v obsresults,linresults;
    f2v results;
    int i,ii,j,jj,k,kk,order0=0,order1, p,q,r,n,nn;
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
    if(obssel.all)
    {
         nn=n;
    }
    else
    {
        nn=obssel.list.size();
    }
    if(nn==0) return results;
    for (ii=0;ii<nn;ii++)
    {
        if(obssel.all)
        {
             i=ii;
        }
        else
        {
             i=obssel.list(ii);
        }
        r=data[i].offset.n_elem;
        q=data[i].indep.n_cols;
        lambda.set_size(r);
        lambda=data[i].offset;
        if(order>0&&q>0)obsresults.grad.set_size(r);
        if(order>1&&q>0)obsresults.hess.set_size(r,r);
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
        if(q==0||order==0)
        {
            addsel(order0,data[i].xselect,obsresults,results,data[i].weight);
        }
        else 
        {
            if(order>2)obsresults.hess=-obsresults.grad*obsresults.grad.t();
            linresults.grad.set_size(q);
            if(order>1) linresults.hess.set_size(q,q);
            linresults=linsel(order,obsresults,data[i].indep);
            addsel(order,data[i].xselect,linresults,results,data[i].weight);  
        }
    }
    return results;
}
