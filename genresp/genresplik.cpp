//Log likelihood and its gradient and Hessian
//for response model with responses data.
//Model codes are found in genresp.cpp.
//If order is 0, only the function is
//found, if order is 1, then the function and gradient are found.  If order is 2,
//then the function, gradient, and Hessian are returned.   If order is 3,
//then the Hessian is replaced by the approximate Hessian
//of the Louis method.
//patterns indicates full specification of models used for responses.
//patternnumber connects data to model pattern.
//selectbeta connects beta elements to matrix x of patterns.
//selectbetano connects data to selectbeta elements.
//Weight w is used.
//obssel indicates which observations to consider.
//beta is parameter vector.
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
//all is for selection of all elements (true) or some or no elements (false).
//If all is false, list indicates element numbers to use.
struct xsel
{
    bool all;
    uvec list;
};
//Specify a model.
//choice is model distribution.
//o is constant vector.
//x is tranformation from beta elements used to lamnbda that does not involve theta.
//c is transformation from beta elements used and theta double elements  used to lambda.
struct pattern
{
    model choice;
    vec o;
    mat x;
};
f2v genresp(const int & , const model &, const resp &, const vec &);
void addsel(const int & , const xsel & , const f2v & , f2v & , const double & );
f2v linsel(const int & , const f2v & , const mat & );
int intsel(const xsel & , const int & );
ivec ivecsel(const xsel & , const ivec & );
vec vecsel(const xsel & , const vec & );
int sintsel(const xsel & , const int & );
int sivecsel(const xsel & , const ivec & );
int svecsel(const xsel & , const vec & );
f2v genresplik(const int & order, const field<pattern> & patterns,
    const xsel & patternnumber, const field<resp> & data,
    const field<xsel> & selectbeta, const xsel & selectbetano,
    const vec & w, const xsel & obssel, const vec & beta)
{
//Linear model parameter is gamma.
//lambda is final model parameter.
//t is theta component used in lambda quadratic term.
    vec gamma, lambda, t;
//Matrix of all interactions of lambda and gamma.
    mat indep;
//Log likelihood, gradient, and Hessian.
    f2v obsresults, linresults, results;
//Counters and sizes.
//i locates observations,
//ii counts observations.
//j locates patterns.
//jj locates beta selections.
//k is dimension of lambda.
//n is total observation count.
//nn counts observations  used.
//order0 is for only values.
//order1 is for values and gradients.
//p is dimension of beta.
//q is dimension of used part of beta.
    int i, ii, j, jj, k, kk, n, nn, order0 = 0,
        order1, p, q, qq;
//Starting value for log likelihood.
    results.value = 0.0;
//Number of elements of beta.
    p=beta.n_elem;
//Number of observations.
    n=data.n_elem;
//Set up results elements.
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
//Order specification if Louis approach used.
    order1=order;
    if(order>2)order1=1;
//Select observations.
    nn=sintsel(obssel,n);
//If nothing selected, nothing to do.
    if(nn==0) return results;
//Cycle through observations.
    for (ii=0;ii<nn;ii++)
    {
        i=intsel(obssel,ii);
//Pattern to use.
        j=intsel(patternnumber,i);
//Beta selection to use.
        jj=intsel(selectbetano,i);
//Now for predictors.
        k=patterns(j).o.n_elem;
        lambda.set_size(k);
        lambda=patterns(j).o;
        q=patterns(j).x.n_cols;
        if(q>0)
        {
            gamma.set_size(q);
            gamma=vecsel(selectbeta(jj),beta);
            lambda=lambda+patterns(j).x*gamma;
        }
        if(order>0&&q>0)obsresults.grad.set_size(q);
        if(order>1&&q>0)obsresults.hess.set_size(q,q);
        if(q>0)
        {    
            obsresults=genresp(order1, patterns(j).choice, data(i), lambda);
        }
        else
        {
            obsresults=genresp(order0, patterns(j).choice, data(i), lambda);
        }
        if(isnan(obsresults.value))
        {
            results.value=datum::nan;
            if(order>0)results.grad.fill(datum::nan);
            if(order>1)results.hess.fill(datum::nan);
            return results;
        }
        if(order>0)
        {
            if(!results.grad.is_finite())
            {
                results.value=datum::nan;
                results.grad.fill(datum::nan);
                if(order>1)results.hess.fill(datum::nan);
                return results;
            }
        }
        if(order>1)
        {
            if(!results.hess.is_finite())
            {
                results.value=datum::nan;
                results.grad.fill(datum::nan);
                results.hess.fill(datum::nan);
                return results;
            }
        }
        if(q==0||order==0)
        {
            addsel(order0,selectbeta(jj),obsresults,results,w(i));
        }
        else 
        {
            if(order>2)obsresults.hess=-obsresults.grad*obsresults.grad.t();
            linresults.grad.set_size(q);
            if(order>1) linresults.hess.set_size(q,q);
            linresults=linsel(order,obsresults,patterns(j).x);
            addsel(order,selectbeta(jj),linresults,results,w(i));
        }
    }
    return results;
}
