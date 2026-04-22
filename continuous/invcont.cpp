//Inverse transformation at location and log scale parameters
//corresponding to expectation m and standard deviation s for
//minimum Gumbel (transform='G'), maximum Gumbel (transform='H'),
//logistic (transform='L'), and normal (transform='N').
#include<armadillo>
#define STATS_ENABLE_ARMA_WRAPPERS
#include "stats.hpp"
using namespace arma;

vec invcont(const char & transform, const double & m, const double & s)
{
    vec result(2);
//Euler constant.
    double gamma=0.577215664901533;
    if(s<=0.0){
        result(0)=datum::nan;
        result(1)=datum::nan;
        return result;
    }
    switch (transform)
    {
        case 'G':
            result(1)=sqrt(6.0)*s/datum::pi;
            result(0)=m-gamma*result(1);
            result(1)=log(result(1));
            return result;

        case 'H':
            result(1)=sqrt(6.0)*s/datum::pi;
            result(0)=m+gamma*result(1);
            result(1)=log(result(1));
            return result;

        case 'N':
            result(0)=m;
            result(1)=log(s);
            return result;

        default:
            result(0)=m;
            result(1)=log(sqrt(3.0)*s/datum::pi);
            return result;
    }
}
