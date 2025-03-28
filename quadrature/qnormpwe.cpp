//Compute points and weights for normal scores quadrature of given order.  Use conditional 
//expectations.
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
pw qnormpwe(const int & n)
{
    double x,xn,y,z;
    int i,n1;
    xn=1.0/double(n);
    n1=n-1;
    pw pws;
    pws.points.set_size(n);
    pws.weights.set_size(n);
    pws.weights.fill(xn);
    x=xn;
    for(i=0;i<n;i++)
    {
        if(i<n1)y=normpdf(stats::qnorm(x,0.0,1.0));
        if(i==0)
        {
            pws.points(0)=-y/xn;
        }
        else
        {
            if(i==n1)
            {
                pws.points(n1)=z/xn;
                break;
            }
            else
            {
                pws.points(i)=(z-y)/xn;
            }
        }
        z=y;
        x=x+xn;
    }
    pws.points=pws.points/stddev(pws.points,1);
    return pws;
}
            
            
            
