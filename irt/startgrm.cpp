//Starting values for graded response model with a standard normal
//latent variable.  The elementary case is considered in which all individuals
//have the same items and responses are nonnegative integers.
//responses is the data matrix.  cdf is the distribution specification, with
//'G' for minimum Gumbel, 'H' for maximum Gumbel, 'L' for logistic, and 'N' for normal.


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
struct f1v
{
    vec values;
    double info; 
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
f1v invgrad(const char & , const vec & probs);
vec startgrm(const int & order , const params & mparams, const char & algorithm, 
    const char & cdf, const imat & responses)
{
    int d, i, j, k, n, p, r;
    double q=0.0,x,y;
    n=responses.n_rows;
    x=double(n);
    p=responses.n_cols;
    r=p*(p-1)/2;
    field<f1v>tranm(p);
    field<resp>yy(r);
    irowvec nmax(p);
    nmax=max(responses,0);
    d=p+sum(nmax);
    field<ivec>counts(p);
    field<vec>probs(p);
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
         probs(j).set_size(nmax(j));
         probs(j)=(conv_to<vec>::from(counts(j).tail(nmax(j))))/x;
         tranm(j).values.set_size(nmax(j));
         tranm(j)=invgrad(cdf,probs(j));
         results.subvec(k,k+nmax(j)-1)=tranm(j).values;
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
              yy(i).dresp(0)=c(j,k)/(tranm(j).info*tranm(k).info);
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
