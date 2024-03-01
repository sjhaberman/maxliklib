//Starting values for one-parameter IRT model with a standard normal
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
vec startoneparamirt(const char & cdf, const imat & responses)
{
    int d, j, k, p, r, n;
    double q=0.0, s=0.0;
    n=responses.n_rows;
    p=responses.n_cols;
    d=p+1;
    r=p*(p-1)/2;
    vec results(d), start(p);
    
    mat m(1,p),rsp(n,p),c(p,p);
    rsp=conv_to<mat>::from(responses);
    field<f1> tranm(p);
    m=mean(rsp);
//No solution.
    if(m.min()==0.0||m.max()==1.0)
    {
         results.fill(datum::nan); 
         return results;
    }
    c=cov(rsp);
    for(j=1;j<p;j++)
    {
         for(k=0;k<j;k++)
         {
              q=q+c(j,k);
         }
    }
    q=q/(double)r;    
    for(j=0;j<p;j++)
    {
         tranm(j)=invcdf(cdf,m(0,j));
         results(j)=tranm(j).value;
         s=s+tranm(j).der;
    }
    results(p)=q*s/(double)p;
    return results;
}
