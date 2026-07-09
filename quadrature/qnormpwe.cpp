//Compute points and weights for normal scores quadrature of given order.
//Use conditional expectations. Order is n.
#include<armadillo>
#define STATS_ENABLE_ARMA_WRAPPERS
#include "stats.hpp"
using namespace arma;
struct pw{vec points; vec weights;};
pw qnormpwe(const uword & n){
    double x,xn,xu,y,z;
    vec::iterator pit;
    xn=1.0/double(n);
    xu=1.0-0.5*xn;
    pw pws;
    pws.points.set_size(n);
    pws.weights.set_size(n);
    pws.weights.fill(xn);
    x=xn;
    for(pit=pws.points.begin();pit<pws.points.end();++pit){
        if(x<=xu)y=normpdf(stats::qnorm(x));
        if(pit==pws.points.begin()) *pit=-y/xn;
        else{
            if(x>xu){
                *pit=z/xn;
                break;
            }
            else *pit=(z-y)/xn;
        }
        z=y;
        x+=xn;
    }
    pws.points=pws.points/stddev(pws.points,1);
    return pws;
}
            
            
            
