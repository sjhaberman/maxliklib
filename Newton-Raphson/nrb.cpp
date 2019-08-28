//Newton-Raphson algorithm for function maximization on an open interval
//for a strictly concave function that approaches negative infinity as
//absolute value of argument goes to infinity or as value of argument goes to finite
//upper bound or finite lower bound of interval.  The bool variable f.fin is true if the
//argument is in the interval.  Otherwise f.fin is false.
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
struct fd2b
{
    double value;
    double der1;
    double der2;
    bool fin;
};

struct nrvarb
{
    double locmax;
    double max;
    double der1;
    double der2;
    bool fin;
};
nrvarb nrvarbf(double x,function <fd2b(double)> f)
{
    fd2b resultf;
    nrvarb result;
    resultf=f(x);
    result.fin=resultf.fin;
    result.locmax=x;
    if (resultf.fin)
    {
        result.max=resultf.value;
        result.der1=resultf.der1;
        result.der2=resultf.der2;
    }
    return result;
}
double davidon(double,double,double,double,double,double);
double newton(double,double,double);
double modit(double,double,double,double,double);
void rebound(double,double,double &,double &);
nrvarb nrb(const int maxit,const double tol,const double start,
         const double stepmax,const double b,
         function<fd2b(double)> f)
{
    
    
    double d,deltaf,lower,upper,x,y;
    int i;
    nrvarb varx,vary;
// x is old value.
    x=start;
// lower is lower bound for location of maximum.
    lower=-DBL_MAX;
// upper is upper bound for location of maximum.
    upper=DBL_MAX;
    
// varx is current location, function, first derivative, and second derivative.
    varx=nrvarbf(x,f);
// Stop if derivative is 0 or starting value is inadmissible.
    if(!varx.fin) return varx;
    if(varx.der1==0.0) return varx;
    rebound(x,varx.der1,lower,upper);
// Up to maxit iterations.
    for(i=0;i<maxit;i++)
    {
// The proposed step.  Keep within range.
        y=newton(x,varx.der1,varx.der2);

        y=modit(x,y,stepmax,lower,upper);
// New bounds.
// Get new function value, new derivative, and new second derivative.
        vary=nrvarbf(y,f);
        while(!vary.fin)
        {
            y=0.5*(x+y);
            vary=nrvarbf(y,f);
        }
        if(vary.der1==0.0)return vary;
        rebound(y,vary.der1,lower,upper);
        

// Look for adequate progress.
        d=y-x;
        deltaf=vary.max-varx.max;
        if(d!=0.0&&deltaf>=b*fabs(d*vary.der1))
        {
// Check for convergence.
            if(deltaf<tol) return vary;
        }
// Treat inadequate progress.
        else
        {
// See if sign of first derivative has changed.
            if(d*vary.der1<0.0)
            {
// Cubic interpolation.
                y=davidon(x,y,varx.locmax,vary.locmax,varx.der1,vary.der1);
                vary=nrvarbf(y,f);
                if(vary.der1==0.0)return vary;
                rebound(y,vary.der1,lower,upper);

            }
        }
       
        x=y;
        varx=vary;
    }
    return vary;
}

