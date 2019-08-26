//Log likelihood component and its gradient and hessian matrix
//for location and scale model for base distribution with log-concave density with
//negative second derivative of the logarithm f of the density.  Model has
//density f(y) = g(beta(1)*y+beta(0))beta(1), where g is the density of the
//standard version of the distribution.
//The convention is used that beta(1)>0.

#include<armadillo>
using namespace arma;
using namespace std;
struct fd2
{
    double value;
    double der1;
    double der2;
};
struct fd2bv
{
    double value;
    vec grad;
    mat hess;
    bool fin;
};


fd2bv locationscale(double y,vec beta,function <fd2(double)> f)
{
    double z;
    fd2 result;
    
    fd2bv results;
    if(beta(1)<=0.0)
    {
        results.fin=false;
    }
    else
    {
        z=beta(0)+beta(1)*y;
        result=f(z);
        
        results.fin=true;
        
        results.value=result.value+log(beta(1));
        results.grad={result.der1,result.der1*y+1.0/beta(1)};
        results.hess={{1.0,y},{y,y*y}};
        results.hess=result.der2*results.hess;
        results.hess(1,1)=results.hess(1,1)-1.0/(beta(1)*beta(1));
        
    }
    
    return results;
}
