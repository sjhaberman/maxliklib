//Starting values for two-parameter IRT model with a standard normal
//latent variable.  The elementary case is considered in which all individuals
//have the same items and no data are
//missing.  In addition, all observations have unit weight.  Responses are all 0 or 1.
//responses is the data matrix.  cdf is the distribution specification, with
//'G' for gumbel, 'L' for logistic, and 'N' for normal.


#include<armadillo>
using namespace std;
using namespace arma;
struct f1
{
    double value;
    double der; 
};
f1 invcdf(const char & , const double & prob);
struct maxf2v
{
    vec locmax;
    double max;
    vec grad;
    mat hess;
};
struct resp
{
  ivec iresp;
  vec dresp;
};
//Parameters for function maximization.
struct params
{
    bool print;
    int maxit;
    int maxits;
    double eta;
    double gamma1;
    double gamma2;
    double kappa;
    double tol;
};
maxf2v regprod(const int & , const params & , const char & ,
    const field<resp> & , const vec & );
vec starttwoparamirt(const int & order , const params & mparams, const char & algorithm, 
    const char & cdf, const imat & responses)
{
    int d, i, j, k, p, r, n;
    double q=0.0;
    n=responses.n_rows;
    p=responses.n_cols;
    d=p+p;
    r=p*(p-1)/2;
    vec results(d), start(p);
    maxf2v result;
    result.locmax.set_size(p);
    result.grad.set_size(p);
    if(order>1)result.hess.set_size(p,p);
    mat m(1,p),rsp(n,p),c(p,p);
    rsp=conv_to<mat>::from(responses);
    field<f1> tranm(p);
    field<resp> y(r);
    m=mean(rsp);
//No solution.
    if(m.min()==0.0||m.max()==1.0)
    {
         results.fill(datum::nan); 
         return results;
    }
    c=cov(rsp);
    i=0;
    start=ones(p);
    for(j=1;j<p;j++)
    {
         for(k=0;k<j;k++)
         {
              y(i).iresp.set_size(2);
              y(i).iresp(0)=j;
              y(i).iresp(1)=k;
              y(i).dresp.set_size(1);
              y(i).dresp(0)=c(j,k);
              q=q+c(j,k);
              i=i+1;
         }
    }
    q=q/(double)r;
    start=q*start;
    result=regprod(order, mparams, algorithm, y, start);     
    j=0;
    for(i=0;i<p;i++)
    {
         tranm(i)=invcdf(cdf,m(0,i));
         results(j)=tranm(i).value;
         results(j+1)=result.locmax(i)*tranm(i).der;
         j=j+2;
    }
    return results;
}
