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
    int d, i, j, k, p, r, n;
    n=responses.n_rows;
    p=responses.n_cols;
    d=p+1;
    r=p*(p-1)/2;
    vec results(d), start(p),y(r); 
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
    for(i=0;i<p;i++)
    {
         tranm(i)=invcdf(cdf,m(0,i));
         results(i)=tranm(i).value;
    }
    c=cov(rsp,1);
//Predicted transformed covariances.
    i=0;
    for(j=1;j<p;j++)
    {
         for(k=0;k<j;k++)
         {
              y(i)=c(j,k)*tranm(j).der*tranm(k).der;
              i=i+1;
         }
    }
    results(p)=mean(y);
    return results;
}
