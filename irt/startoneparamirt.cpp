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
    double q=0.0, s=0.0;
    n=responses.n_rows;
    p=responses.n_cols;
    d=p+1;
    r=p*(p-1)/2;
    vec results(d), start(p);
    
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
    for(i=0;i<p;i++)
    {
         tranm(i)=invcdf(cdf,m(0,i));
         results(j)=tranm(i).value;
    }
    c=cov(rsp);
//Predicted transformed covariances.
    i=0;
    for(j=1;j<p;j++)
    {
         for(k=0;k<j;k++)
         {
              y(i).iresp.set_size(2);
              y(i).iresp(0)=j;
              y(i).iresp(1)=k;
              y(i).dresp.set_size(1);
              y(i).dresp(0)=c(j,k)*tranm(j).der*tranm(k).der;
              q=q+y(i).dresp(0);
              i=i+1;
         }
    }
    q=sqrt(q/(double)r);
 //Return results. 
    for(j=0;j<p;j++)
    {
         results(j)=tranm(j).value;
    }
    results(p)=q;
    return results;
}
