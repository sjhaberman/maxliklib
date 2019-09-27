//Log likelihood and derivatives for trivial probit case.

#include<armadillo>
using namespace arma;
struct fd2
{
    double value;
    double der1;
    double der2;
};
extern double pvalue;

fd2 normf(double beta)
{
    double p,q,r,s,w;
    fd2 results;
    
    p=normcdf(beta);
    q=1.0-p;
    r=normpdf(beta);
    w=p*q;
    s=r/w;
    results.value=pvalue*log(p)+(1.0-pvalue)*log(q);
    results.der1=s*(pvalue-p);
    results.der2=-(beta+results.der1)*results.der1-pvalue*(1.0-pvalue)*s*s;
    return results;
}
