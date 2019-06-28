//Log likelihood component and its first and second derivative
//for probit model with response y and parameter beta.

#include<armadillo>
using namespace arma;
struct fd2
{
    double value;
    double der1;
    double der2;
};


fd2 probit(int y,double beta)
{
    double p,q,r;
    fd2 results;
    
    p=normcdf(beta);
    q=1.0-p;
    r=normpdf(beta);
    if(y==1)
    {
        results.value=log(p);
        results.der1=r/p;
        results.der2=-(beta+results.der1)*results.der1;
    }
    else
    {
        results.value=log(q);
        results.der1=-r/q;
        results.der2=-(beta+results.der1)*results.der1;
    }
    return results;
}
