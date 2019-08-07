//Newton-Rapshon algorithm for function maximization.
//This function is concerned with maximization along a line for a real function
//on a nonempty open convex subset of a finite-dimensional vector space.
//The function input involves a proposed vector step v from a vector x.
//The function is specified in the struct fd2b with elements value, grad (for gradient),
//hess (for Hessian matrix), and fin (true if argument in set and false otherwise).
//The function f.value(x+av) is maximized approximately over real numbers a such that
//x+av is in the convex set.
//The dot product of v and f.grad(x) is assumed positive.
//The Hessian is always negative definite, and it is assumed that the function approaches
//negative infinity whenever the norm of the argument approaches infinity.
//Maximum number of iterations is maxit.
//The improvement check is b>1.
//The final step is y.  The program uses the struct nrvvarb with elements locmax, max, grad,
//hess, and fin.  For a vector x, locmax is x, max is f.value(x),
//grad is the value of the gradient at x,
//hess is the value of the Hessian at x, fin is true if x is in the set and false otherwise.
//See Chapter 3 of Analysis of Frequency Data by S. J. Haberman for a very closely-related
//algorithm.
//This algorithm works with nrv.cpp.
#include<armadillo>
#include<float.h>
using namespace std;
using namespace arma;
struct fd2bv
{
    double value;
    vec grad;
    mat hess;
    bool fin;
};

struct nrvvarb
{
    vec locmax;
    double max;
    vec grad;
    mat hess;
    bool fin;
};
nrvvarb nrvvarbf(vec x,function<fd2bv(vec)> f);

double davidon(double,double,double,double,double,double);
double newton(double,double,double);
double modit(double,double,double,double,double);
void rebound(double,double,double &,double &);
nrvvarb nrbwv(const int maxit,
         const double stepmax,const double b,vec v,nrvvarb varx,
         function<fd2bv(vec)> f)
{
    
    
    double d,dirx,diry,dirz,dir2x,dir2y,dir2z,fx,fy,fz,lower,upper,xx,yy,zz;
    int i;
    nrvvarb vary,varz;
    if(!varx.fin)return varx;
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
        
        vary=nrvvarbf(varx.locmax+yy*v,f);
        while(!vary.fin)
        {
            yy=0.5*yy;
            vary=nrvvarbf(varx.locmax+yy*v,f);
        }
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
            vary=nrvvarbf(varx.locmax+yy*v,f);
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

