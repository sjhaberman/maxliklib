//Starting values for two-parameter IRT model with a standard normal
//latent variable.  The elementary case is considered in which all individuals
//have the same items and no data are
//missing.  In addition, all observations have unit weight.  Responses are all 0 or 1.
//They are obtained from a comma-separated file with no other entries.
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
vec starttwoparamirt(const char & cdf, const imat & responses)
{
    int d, i, j=0, n, p;
    double q;
    n=responses.n_rows;
    p=responses.n_cols;
    d=p+p;
    vec results(d),eval(p);
    mat m(1,p),rsp(n,p),c(p,p),evec(p,p);
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
    eig_sym(eval,evec,c);
    q=sqrt(eval(p-1));
    for(i=0;i<p;i++)
    {
         tranm(i)=invcdf(cdf,m(0,i));
         results(j)=tranm(i).value;
         results(j+1)=q*evec(i,p-1)*tranm(i).der;
         j=j+2;
    }
    return results;
}
