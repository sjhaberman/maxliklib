//Log of integrand and derivative and second derivative
//with respect to latent variable
//for two-parameter IRT model with
//p responses.  No covariates are used.  The latent variable
//has a standard normal distribution.
#include<armadillo>
using namespace arma;


struct fd2
{
    double value;
    double der1;
    double der2;
};

extern vec beta;
extern char choices[];
extern char choice;
extern ivec iy;

fd2 quantal(int,double);
fd2 normal012(double);

fd2 irt2ptheta(double theta)
{
    fd2 results;
    fd2 itemresults;
    int i;
    double intercept,slope;
    results=normal012(theta);
    for(i=0;i<iy.n_elem;i++)
    {
        intercept=beta[2*i];
        slope=beta[2*i+1];
        choice=choices[i];
        itemresults=quantal(iy[i],intercept+slope*theta);
        results.value=results.value+itemresults.value;
        results.der1=results.der1+slope*itemresults.der1;
        results.der2=results.der2+slope*slope*itemresults.der2;
    }
    
    return results;
}

