//Adjustments of eigenvectors of correlation matrix for
//initial slopes in 2-parameter quantal model with standard
//normal latent variable and no covariates.
//Probability is p and beta is corresponding parameter.
#include<armadillo>
#include<cmath>
using namespace arma;
using namespace std;
extern char choice;



double dquant(double p,double beta)
{
    switch (choice)
    {
        case 'C': return -sqrt((1.0-p)/p)*log(1.0-p);
        case 'L': return sqrt(p*(1.0-p));
        case 'M': return sqrt(p);
        default: return normpdf(beta)/sqrt(p*(1.0-p));
    }
}
