//Newton-Rapshon algorithm for function maximization.
//Function and first two derivatives are f, der1, and der2<0,
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
struct fd2
{
    double value;
    double der1;
    double der2;
};

struct nrvar
{
    double locmax;
    double max;
    double der1;
    double der2;
};
nrvar nrvarf(double x,function <fd2(double)> f)
{
    fd2 resultf;
    nrvar result;
    resultf=f(x);
    result.locmax=x;
    result.max=resultf.value;
    result.der1=resultf.der1;
    result.der2=resultf.der2;
    return result;
}
double davidon(double,double,double,double,double,double);
double newton(double,double,double);
double modit(double,double,double,double,double);
void rebound(double,double,double &,double &);
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
    rebound(x,varx.der1,lower,upper);
// Up to maxit iterations.
    for(i=0;i<maxit;i++)
    {
// The proposed step.  Keep within range.
        y=newton(x,varx.der1,varx.der2);

        y=modit(x,y,stepmax,lower,upper);
// New bounds.
        


// Get new function value, new derivative, and new second derivative.
        vary=nrvarf(y,f);
// Stop for 0 derivative.
        if(vary.der1==0.0)return vary;
        rebound(y,vary.der1,lower,upper);

// Look for adequate progress.
        d=y-x;
        deltaf=vary.max-varx.max;
        if(deltaf>=b*fabs(d*vary.der1))
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
                vary=nrvarf(y,f);
                if(vary.der1==0.0)return vary;
                rebound(y,vary.der1,lower,upper);

            }
        }
       
        x=y;
        varx=vary;
    }
    return vary;
}

