//Newton-Rapshon algorithm for function maximization.
//Function and first two derivatives are f.value, f.der1, and f.der2<0,
//Maximum number of iterations is maxit.
//If change in approximation to maximum is less
//than tol, then iterations cease.
//The largest permitted step is stepmax>0.
//The improvement check is b>1.
//See Chapter 3 of Analysis of Frequency Data by S. J. Haberman for a very closely-related
//algorithm.
#include<cstdlib>
#include<functional>
#include<cmath>
#include<float.h>
using namespace std;
#include "nr.h"
nrvar nr(const int maxit,const double tol,const double start,
         const double stepmax,const double b,
         function<fd2(double)> f)
{
    
    
    double d,deltaf,lower,upper,x,y;
    int i;
    nrvar varx,vary;
// x is old value.
    x=start;
// lower is lower bound for location of maximum.
    lower=-DBL_MAX;
// upper is upper bound for location of maximum.
    upper=DBL_MAX;
    
// varx is current location, function, first derivative, and second derivative.
    varx=nrvarf(x,f);
// Stop if derivative is 0.
    if(varx.der1==0.0) return varx;
// Up to maxit iterations.
    for(i=0;i<maxit;i++)
    {
// The proposed step.
        d=fmax(-stepmax,fmin(stepmax,-varx.der1/varx.der2));

// The proposed new value.
        y=x+d;
// The case of a
        if(y>x)
        {
            lower=x;
            if(y>upper)y=upper;
        }
        else
        {
            upper=x;
            if(x<lower)y=lower;
        }
// Update d in case y was modified.
        d=y-x;
// Get new function value, new derivative, and new second derivative.
        vary=nrvarf(y,f);
// Stop for 0 derivative.
        if(vary.der1==0.0) return vary;

// Look for adequate progress.
        deltaf=vary.max-varx.max;
        if(deltaf>=b*fabs(d*vary.der1))
        {
// Check for convergence.
            if(deltaf<tol) return vary;
        }
// Treat inadequate progress.
        else
        {
            if(vary.der1<0.0)
            {
                if(y>x)
                {
                    upper=y;
                }
                else
                {
                    lower=y;
                }
                
                y=x+d*varx.der1/(2.0*(varx.der1-deltaf/d));
                vary=nrvarf(y,f);
            }
        }
       
        x=y;
        varx=vary;
    }
    return vary;
}

