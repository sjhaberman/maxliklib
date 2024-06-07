//Inverse graded response cdf and derivatives for Gumbel, logistic, and normal cases.
//cdf is 'G'for Gumbel, 'L' for logistic, and 'N' for normal.
#include<armadillo>
#define STATS_ENABLE_ARMA_WRAPPERS
#include "stats.hpp"
using namespace arma;

struct f1v
{
    vec values;
    double info;
};
f1v invgrad(const char & cdf, const vec & probs)
{     
    f1v result;
    int i, p;
    p=probs.n_elem;
    result.values.set_size(p);
    vec cs(p),ders(p);
    cs=reverse(cumsum(reverse(probs)));
    double q;
    for(i=0;i<p;i++)
    {   
         switch (cdf)
         {
            case 'G':
                 q=1.0-cs(i);
                 result.values(i)=log(-log(q));
                 ders(i)=q*exp(result.values(i));
                 continue;

            case 'H':
                 result.values(i)=-log(-log(cs(i)));
                 ders(i)=cs(i)*exp(-result.values(i));
                 continue;

            case 'N':
                 result.values(i)=stats::qnorm(cs(i),0.0,1.0);
                 ders(i)=normpdf(result.values(i));
                 continue;

            default:
                 q=1.0-cs(i);
                 result.values(i)=log(cs(i)/q);
                 ders(i)=cs(i)*q;
                 continue;
         }
    }
    result.info=ders(0)*ders(0)/(1.0-cs(0))+ders(p-1)*ders(p-1)/probs(p-1);
    if(p>1)
    {
         for(i=1;i<p;i++)
         {
              q=ders(i)-ders(i-1);
              result.info=result.info+q*q/probs(i-1);
         }
    }
    return result;
}
              