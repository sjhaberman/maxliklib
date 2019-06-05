//Newton-Raphson algorithm for function maximization for a real function
//on the space real vectors of some fixed finite dimension.
//The function, gradient, and Hessian are
//f.value, f.grad, and f.hess.
//The Hessian is assumed continuous.
//The function must approach negative infinitely whenever the vector norm of the function argument
//approaches infinity, and there must be a single critical point.
//Maximum number of main iterations is maxit.
//Maximum number of secondary iterations per main iteration is maxits.
//If change in approximation to maximum is less
//than tol, then iterations cease.
//The largest permitted step is stepmax>0
//The improvement check is b>1.
//The minimum ratio of inner product of tentative step and gradient versus inner product of
//gradient and gradient must at least be mult>0.
//See Chapter 3 of Analysis of Frequency Data by S. J. Haberman for a very closely-related
//algorithm.


#include<float.h>
#include<armadillo>
using namespace std;
using namespace arma;
#include "nrv.h"

nrvvar nrvs(const int maxit,const int maxits,const double tol,const vec start,
        const double stepmax,const double b,const double mult,
        function<fd2v(vec)> f)
{
    
    
    bool sing;
    double c,cc,d,dd,deltaf,dirx,diry,dir2y,dirz,dir2z,ee,lower,upper,steplim;
    int i,j;
    nrvvar varx,vary,varz;
    vec v,x,y,z;
// Old value of location is x.
    x=start;
   
// Function settings.
    varx=nrvvarf(x,f);
// Stop if gradient of zero.
    if(!any(varx.grad)) return varx;
    
// Iterations.
    for(i=0;i<maxit;i++)
    {
// The basic step.
        sing=solve(v,-varx.hess,varx.grad);
// Revert to steepest ascent.
        if(sing)
        {
            v=varx.grad;
        }
        else
        {
            d=dot(v,varx.grad)/dot(varx.grad,varx.grad);
            if(d<mult)v=varx.grad;
        }
        d=norm(v,"inf");

// Step truncation.
        if(d<=stepmax)
        {
            c=1.0;
        }
        else
        {
            c=stepmax/d;
        }
        v=c*v;
// Tentative new value.
        y=x+v;
// New value, gradient, and hessian.
        vary=nrvvarf(y,f);
// Stop if gradient of zero.
        if(!any(vary.grad)) return vary;
// Look for adequate progress.
       
        deltaf=vary.max-varx.max;
        
        diry=dot(v,vary.grad);
        
        if(deltaf>=b*fabs(diry))
        {
// Check for convergence.
            if(deltaf<tol) return vary;
        }
// Treat inadequate progress.
        else
        {
            lower=0.0;
            upper=DBL_MAX;
            cc=1.0;
            dir2y=dot(v,vary.hess*v);
            
// Up to maxita iterations.
            for(j=0;j<maxits;j++)
            {
// The proposed step.
                if(dir2y<0.0)
                {
                    dd=fmax(-stepmax,fmin(stepmax,-diry/dir2y));
                }
                else
                {
                    if(diry<0.0)
                    {
                        dd=-stepmax;
                    }
                    else
                    {
                        dd=stepmax;
                    }
                    
                }
                
                
                
// The proposed new value.
                ee=cc+dd;
            
// New bounds.
                if(dd>0.0)
                {
                    lower=cc;
                    if(ee>upper)ee=upper;
                }
                else
                {
                    upper=cc;
                    if(ee<lower)ee=lower;
                }
// Update dd  in case of truncation.
                dd=ee-cc;
// Get new function value, new derivative, and new second derivative.
                z=x+ee*v;
                varz=nrvvarf(z,f);
                dirz=dot(v,varz.grad);
                dir2z=dot(v,varz.hess*v);
// Stop for 0 derivative.
                if(dirz==0.0) break;
                
// Look for adequate progress.
                deltaf=varz.max-varx.max;
                if(deltaf>=b*fabs(dirz)) break;

// Treat inadequate progress.
                if(dirz<0.0)
                {
                    if(ee>cc)
                    {
                        upper=ee;
                    }
                    else
                    {
                        lower=ee;
                    }
                    deltaf=varz.max-vary.max;
                    ee=cc+dd*diry/(2.0*(diry-deltaf/dd));
                    y=x+ee*v;
                    vary=nrvvarf(y,f);
                }
            }
            varx=varz;
            
        }
            
        varx=vary;
        x=varx.locmax;
    }
    return vary;
}

