//Compute points and weights for normal scores quadrature of given order.
//Order is n.
#include<armadillo>
#define STATS_ENABLE_ARMA_WRAPPERS
#include "stats.hpp"
using namespace arma;
struct pw
{
    vec points;
    vec weights;
};
pw qnormpw(const int & n)
{
    double x,xn;
    int i;
    pw pws;
    pws.points.set_size(n);
    pws.weights.set_size(n);
    xn=(double)n;
    for(i=0;i<n;i++)
    {
        x=((double)i+0.5)/xn;
        pws.points(i)=stats::qnorm(x,0.0,1.0);
        pws.weights(i)=1.0/xn;
    }
    pws.points=pws.points/stddev(pws.points,1);
    return pws;
}
            
            
            
