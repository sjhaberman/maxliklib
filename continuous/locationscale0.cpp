//Log likelihood component
//for location and scale model for base distribution with log-concave density with
//negative second derivative of the logarithm f of the density.  Model has
//density f(y) = g(beta(1)*y+beta(0))beta(1), where g is the density of the
//standard version of the distribution.
//The convention is used that beta(1)>0.

#include<armadillo>
using namespace arma;
using namespace std;

struct fd0bv
{
    double value;
    
   
    bool fin;
};


fd0bv locationscale0(double y,vec beta,function <double(double)> f)
{
    double z;
    double result;
    
    fd0bv results;
    if(beta(1)<=0.0)
    {
        results.fin=false;
    }
    else
    {
        z=beta(0)+beta(1)*y;
        result=f(z);
        
        results.fin=true;
        
        results.value=result+log(beta(1));
        
    }
    
    return results;
}
