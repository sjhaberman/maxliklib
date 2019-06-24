//Newton-Rapshon algorithm for function maximization.
//This function is concerned with maximization along a line for a real function
//on a finite-dimensional vector space.
//The function input involves a proposed vector step v from a vector x.
//The function is specified in the struct fd2 with elements value, grad (for gradient),
//and hess (for Hessian matrix. The maximum of f.value(x+av) is maximized approximately over a
//real.  The dot product of v and f.grad(x) is assumed positive.
//The Hessian is always negative definite, and it is assumed that the function approaches
//negative infinity whenever the norm of the argument approaches infinity.
//Maximum number of iterations is maxit.
//The improvement check is b>1.
//The final step is y.  The program uses the struct nrvvar with elements locmax, max, grad,
//and hess.  For vecx, locmax is x, max is f.value(x), grad is the value of the gradient at x,
//and hess is the value of the Hessian at x.  Similar definitions apply to vecy in the case of y.
//See Chapter 3 of Analysis of Frequency Data by S. J. Haberman for a very closely-related
//algorithm.
//This algorithm works with nrv.cpp.
#include<armadillo>
#include<float.h>
using namespace std;
using namespace arma;
struct fd2v
{
    double value;
    vec grad;
    mat hess;
};

struct nrvvar
{
    vec locmax;
    double max;
    vec grad;
    mat hess;
};
nrvvar nrvvarf(vec x,function<fd2v(vec)> f);

double davidon(double,double,double,double,double,double);
double newton(double,double,double);
double modit(double,double,double,double,double);
void rebound(double,double,double &,double &);
nrvvar nrwv(const int maxit,
         const double stepmax,const double b,vec v,nrvvar varx,
         function<fd2v(vec)> f)
{
    
    
    double d,dirx,diry,dirz,dir2x,dir2y,dir2z,fx,fy,fz,lower,upper,xx,yy,zz;
    int i;
    nrvvar vary,varz;

    zz=0.0;
// lower is lower bound for location of maximum.
    lower=0.0;
// upper is upper bound for location of maximum.
    upper=DBL_MAX;
    
// Need value of f.value at x.
    fx=varx.max;
// Current location.
    varz=varx;
    fz=varz.max;
    dirz=dot(v,varz.grad);
    dir2z=dot(v,varz.hess*v);
    

// Up to maxit iterations.
    for(i=0;i<maxit;i++)
    {
// The proposed step.  Keep within range.
        yy=newton(zz,dirz,dir2z);

        yy=modit(zz,yy,stepmax,lower,upper);
// Get new function value, new derivative, and new second derivative.
        
        vary=nrvvarf(varx.locmax+yy*v,f);
        fy=vary.max;
        diry=dot(v,vary.grad);
        dir2y=dot(v,vary.hess*v);
// Stop for 0 derivative.
        if(diry==0.0)return vary;
        rebound(yy,diry,lower,upper);

// Look for adequate progress.
        
        
        if(fy-fx>=b*yy*fabs(diry)) return vary;
        
// Treat inadequate progress.
        
        
// See if sign of first derivative has changed.
        if(diry*dirz<0.0)
        {
// Cubic interpolation.
            yy=davidon(zz,yy,fz,fy,dirz,diry);
            vary=nrvvarf(varx.locmax+yy*v,f);
            diry=dot(v,vary.grad);
            if(diry==0.0)return vary;
            rebound(yy,diry,lower,upper);

        }
//Update for next cycle.
       
        varz=vary;
        zz=yy;
        dirz=diry;
        dir2z=dir2y;
    }
    return vary;
}

