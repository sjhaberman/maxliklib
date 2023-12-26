//Inverse cdf and derivatives for Gumbel, logistic, and normal cases.
//cdf is 'G'for Gumbel, 'L' for logistic, and 'N' for normal.
#include<armadillo>
#define STATS_ENABLE_ARMA_WRAPPERS
#include "stats.hpp"
using namespace arma;
struct f1
{
    double value;
    double der;
};
f1 invcdf(const char & cdf, const double & prob)
{     
    f1 result;
    double q;    
    switch (cdf)
    {
       case 'G':
            q=log(1.0-prob);
            result.value=-log(-q);
            result.der=-1.0/((1.0-prob)*q);
            return result;

       case 'N':
            result.value=stats::qnorm(prob,0.0,1.0);
            result.der=1.0/normpdf(result.value);
            return result;

       default:
            q=1.0-prob;
            result.value=log(prob/q);
            result.der=1.0/(prob*q);
            return result;
    }
}
