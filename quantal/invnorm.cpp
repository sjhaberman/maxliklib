//Inverse cdf of standard normal.
#include<cmath>
#include<functional>
using namespace std;
extern double pvalue;
struct fd2
{
    double value;
    double der1;
    double der2;
};
struct nrvar
{
    double locmax;
    double max;
    double der1;
    double der2;
};
nrvar nr(const int,const double,const double,
    const double,const double,function<fd2(double)>);
fd2 normf(double);
double invnorm(double p)
{
    pvalue=p;
    double stepmax=3.0;
    double tol=0.00000000001;
    double b=1.1;
    double start;
    int maxit=10;
    
    nrvar results;
    start=0.59*log(pvalue/(1.0-pvalue));
    results=nr(maxit,tol,start,stepmax,b,normf);
    
    return results.locmax;
}
