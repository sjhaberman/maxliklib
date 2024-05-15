//Starting values for one-parameter IRT model with a standard normal
//latent variable.  The elementary case is considered in which all individuals
//have the same items and no data are
//missing.  In addition, all observations have unit weight.  Responses are all 0 or 1.
//responses is the data matrix.  cdf is the distribution specification, with
//'G' for minimum Gumbel, 'H' for maximum Gumbel, 'L' for logistic, and 'N' for normal.


#include<armadillo>
using namespace std;
using namespace arma;
struct f1
{
    double value;
    double info; 
};
f1 invcdf(const char & , const double & prob);
struct resp
{
  ivec iresp;
  vec dresp;
};
vec startoneparamirt(const char & cdf, const imat & responses)
{
    double x,z;
    int d, i, j, k, p, r, n;
    n=responses.n_rows;
    p=responses.n_cols;
    d=p+1;
    r=p*(p-1)/2;
    vec results(d), start(p),y(p); 
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
         y(i)=tranm(i).info;
    }
    c=cov(rsp,1);
//Predicted transformed covariances.
    i=0;
    x=sum(y);
    x=x*x-sum(square(y));
    results(p)=sqrt((accu(c)-trace(c))/x);
    return results;
}
