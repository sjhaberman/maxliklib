//Inverse cumulative model for Gumbel, logistic, and normal cases.
//transform is 'G'for minimum Gumbel, 'H' for maximum Gumbel, 'L' for logistic,
//and 'N' for normal.
#include<armadillo>
#define STATS_ENABLE_ARMA_WRAPPERS
#include "stats.hpp"
using namespace arma;
vec invcum(const char & transform, const vec & probs)
{
    vec result;
    int i, p, q;
    p=probs.n_elem;
    result.set_size(p-1);
    q=p-1;
    double r, s, t;
    r=probs(q);
    for(i=q-1;i>=0;i--){
        s=r;
        r=s+probs(i);
        t=s/r;
        switch (transform){
            case 'G':
                result(i)=log(-log(t));
                break;

            case 'H':
                result(i)=-log(-log(1.0-t));
                break;

            case 'N':
                result(i)=stats::qnorm(t,0.0,1.0);
                break;

            default:
                result(i)=log(t/(1.0-t));
        }
    }
    return result;
}
