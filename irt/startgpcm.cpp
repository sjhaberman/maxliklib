//Starting values for generalized partial credit model with a standard normal
//latent variable.  The elementary case is considered in which all individuals
//have the same items and responses are nonnegative integers.
//responses is the data matrix.


#include<armadillo>
using namespace std;
using namespace arma;
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
vec startgpcm(const int & order , const params & mparams, const char & algorithm, 
    const imat & responses)
{
    int d, i, j, k, n, p, r;
    double q=0.0,x,y;
    n=responses.n_rows;
    p=responses.n_cols;
    r=p*(p-1)/2;
    field<resp>yy(r);
    irowvec nmax(p);
    nmax=max(responses,0);
    d=p+sum(nmax);
    field<ivec>counts(p);
    vec results(d), start(p);
    maxf2v result;
    result.locmax.set_size(p);
    result.grad.set_size(p);
    if(order>1)result.hess.set_size(p,p);
    
    k=0;
    for(j=0;j<p;j++)
    {
         counts(j).set_size(nmax(j)+1);
         counts(j).zeros();
         for(i=0;i<n;i++)counts(j)(responses(i,j))=counts(j)(responses(i,j))+1;
         if(min(counts(j))==0)
         {
              results.fill(datum::nan); 
              return results;
         }
         x=double(counts(j)(0));
         for(i=0;i<nmax(j);i++)
         {
              y=double(counts(j)(i+1));
              results(k+i)=log(y/x);
         }
         k=k+nmax(j)+1;  
    }
    mat m(1,p),rsp(n,p),c(p,p);
    rsp=conv_to<mat>::from(responses);
    m=mean(rsp);
    c=cov(rsp,1);
    start=ones(p);
    i=0;
    for(j=1;j<p;j++)
    {
         for(k=0;k<j;k++)
         {
              yy(i).iresp.set_size(2);
              yy(i).iresp(0)=j;
              yy(i).iresp(1)=k;
              yy(i).dresp.set_size(1);
              yy(i).dresp(0)=c(j,k)/(c(j,j)*c(k,k));
              q=q+yy(i).dresp(0);
              i=i+1;
         }
    }
    q=sqrt(q/double(r));
    start=q*start;
    result=regprod(order, mparams, algorithm, yy, start);
    k=0;
    for(j=0;j<p;j++)
    {
         results(k+nmax(j))=result.locmax(j);    
         k=k+nmax(j)+1;  
    }
    return results;
}
