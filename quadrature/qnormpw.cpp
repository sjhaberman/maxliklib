//Compute points and weights for normal scores quadrature of given order.
//Order is n.
#include<armadillo>
#define STATS_ENABLE_ARMA_WRAPPERS
#include "stats.hpp"
using namespace arma;
struct pw{vec points; vec weights;};
pw qnormpw(const uword & n){
    double x,xn;
    vec::iterator pit;
    pw pws;
    pws.points.set_size(n);
    pws.weights.set_size(n);
    xn=1.0/double(n);
    pws.weights.fill(xn);
    x=0.5*xn;
    for(pit=pws.points.begin();pit<pws.points.end();++pit){
        *pit=stats::qnorm(x);
        x+=xn;
    }
    pws.points=pws.points/stddev(pws.points,1);
    return pws;
}
            
            
            
