//Inverse cdf and information for Gumbel, logistic, and normal cases.
//cdf is 'G'for Gumbel, 'L' for logistic, and 'N' for normal.
#include<armadillo>
#define STATS_ENABLE_ARMA_WRAPPERS
#include "stats.hpp"
using namespace arma;
struct f1
{
    double value;
    double info;
};
f1 invcdf(const char & cdf, const double & prob)
{     
    f1 result;
    double q,r,s,t;
    q=1.0-prob;   
    switch (cdf)
    {
       case 'G':
            result.value=log(-log(q));
            result.info=(q/prob)*exp(2.0*result.value);
            return result;
       case 'H':
            result.value=-log(-log(prob));
            result.info=(prob/q)*exp(-2.0*result.value);
            return result;
       case 'N':
            r=stats::qnorm(prob,0.0,1.0);
            result.value=r;
            t=normpdf(r);
            result.info=t*t/(prob*q);
            return result;

       default:
            result.value=log(prob/q);
            result.info=prob*q;
            return result;
    }
}
