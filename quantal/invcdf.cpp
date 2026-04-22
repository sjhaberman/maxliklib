//Inverse cdf for complementary log-log, log-log, logit, and probit cases.
//transform is 'G'for complementary log-log, 'H' for log-log, 'L' for logit,
//and 'N' for probit.
#include<armadillo>
#define STATS_ENABLE_ARMA_WRAPPERS
#include "stats.hpp"
using namespace arma;
vec invcdf(const char & transform, const double & prob)
{
    vec result(1);
    switch (transform)
    {
        case 'G':
            result(0) = log(-log(1.0-prob));
            return result;
        case 'H':
            result(0) = -log(-log(prob));
            return result;
        case 'N':
            result(0) = stats::qnorm(prob,0.0,1.0);
            return result;
        default:
            result(0) = log(prob/(1.0-prob));
            return result;
    }
}
