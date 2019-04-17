//Modified Newton-Rapshon algorithm for function maximization.
//Function and its first two derivatives are f.value, f.der1, and f.der2.  A substitute f.ders<0 for f.der2 is also used.  The maximum number of main iterations is maxit.
//If change in approximation to maximum is less
//than tol, then iterations cease.
//The largest permitted step is stepmax>0.
//The improvement check is b>1.
//The approach here is related to one in an article of S. J. Haberman in Sociological Methodology
//in 1988.
#include<cstdlib>
#include<functional>
#include<cmath>
#include<float.h>
using namespace std;
#include "nrs.h"
nrsvar nrs(const int maxit,const double tol,const double start,
         const double stepmax,const double b,
         function<fd2s(double)> f)
{
    double d,deltaf,lower,upper,x,y;
    int i;
    nrsvar varx,vary;
// lower is lower bound for location of maximum if only one maximum exists.
    lower=-DBL_MAX;
// upper is upper bound for location of maximum if only one maximum exists.
    upper=DBL_MAX;
// x is old value.
    x=start;
// varx is current location, function, first derivative, second derivative, and alternative
// to second derivative.
    varx=nrsvarf(x,f);
// Stop if derivative is 0.
    if(varx.der1==0.0) return varx;
// Up to maxit iterations.
    for(i=0;i<maxit;i++)
    {
// The proposed step.
        if(varx.der2<0.0)
        {
            d=fmax(-stepmax,fmin(stepmax,-varx.der1/varx.der2));
        }
        else
        {
            d=fmax(-stepmax,fmin(stepmax,-varx.der1/varx.der2s));
        }
// The proposed new value.
        y=x+d;
// New bounds.
        if(y>x)
        {
            lower=x;
            if(y>=upper)y=upper;
        }
        if(y<x)
        {
            upper=x;
            if(y<=lower)y=lower;
        }
// Get new function value, new derivative, new second derivative,
// and new alternative to second derivative.
        
        vary=nrsvarf(y,f);
// Stop for 0 derivative.
        if(vary.der1==0.0) return vary;
// Look for adequate progress.
        deltaf=vary.max-varx.max;
        if(deltaf>=b*fabs((y-x)*vary.der1))
        {
// Check for convergence.
            if(deltaf<tol) return vary;
        }
// Treat inadequate progress.
        else
        {
            if((y-x)*vary.der1<0.0)
            {
                if(y>x)
                {
                    upper=y;
                }
                else
                {
                    lower=y;
                }
                y=x+(y-x)*varx.der1/(varx.der1-vary.der1);
                vary=nrsvarf(y,f);
            }
        }
        
        x=y;
        varx=vary;
    }
    return vary;
}


