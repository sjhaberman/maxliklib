//Inverse cumulative model for Gumbel, logistic, and normal cases.
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
f1v invcum(const char & cdf, const vec & probs)
{     
    f1v result;
    int i, p;
    p=probs.n_elem;
    result.values.set_size(p);
    vec cs(p),f(p),rr(p),ders(p);
    double q,r,s,t;
    cs=reverse(cumsum(reverse(probs)));
    r=1.0;
    for(i=0;i<p;i++)
    {
         s=cs(i);
         rr(i)=s/r;   
         switch (cdf)
         {
            case 'G':
                 result.values(i)=log(-log(rr(i)));
                 ders(i)=rr(i)*exp(result.values(i));
                 continue;

            case 'H':
                 q=1.0-rr(i);
                 result.values(i)=-log(-log(q));
                 ders(i)=q*exp(-result.values(i));
                 continue;

            case 'N':
                 result.values(i)=stats::qnorm(rr(i),0.0,1.0);
                 ders(i)=normpdf(result.values(i));
                 continue;

            default:
                 q=1.0-rr(i);
                 result.values(i)=log(q/rr(i));
                 ders(i)=rr(i)*q;
                 continue;
         }
    }
        result.info=ders(0)*ders(0)/(1-cs(p-1))+ders(p-1)*ders(p-1)/probs(p-1);
    if(p>1)
    {
         for(i=1;i<p;i++)
         {
              q=ders(i)-ders(i-1);
              result.info=result.info+q*q/probs(i);
         }
    }
    return result;
}
