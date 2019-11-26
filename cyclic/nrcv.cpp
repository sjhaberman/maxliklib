//Newton-Rapshon algorithm for function maximization for a single coordinate
//of a strictly concave real function of p variable with a continuous and
//negative-definite Hessian matrix.
//The function input involves a coordinate j and a vector x.
//The function is specified in the struct fdc with elements value, der1 (partial
//derivative with respect to element j),
//and der2 (second partial derivative with respect to j). The maximum of
//f.value(x+av) is maximized approximately over a
//real for v the vector with element j equal to 1 and all other elements 0.
//Maximum number of iterations is maxit.
//The improvement check is b>1.
//The final step is y.  The program uses the struct cyclevar with elements
//locmax, max, der1,
//and der2.  For a vector x, locmax is x, max is f.value(x), der1 is the value
//at x of partial derivative j, and der2 is the value at x of second partial
//derivative j.,
//See Chapter 3 of Analysis of Frequency Data by S. J. Haberman
//for a very closely-related
//algorithm.
//This algorithm works with cyclic.cpp.
#include<armadillo>
#include<float.h>
using namespace std;
using namespace arma;
struct fd2
{
    double value;
    double der1;
    double der2;
};

struct cyclevarp
{
    vec locmax;
    double max;
    double der1;
    double der2;
};

cyclevarp cyclevarf(int,vec,function<fd2(int,vec)> );

double davidon(double,double,double,double,double,double);
double newton(double,double,double);
double modit(double,double,double,double,double);
void rebound(double,double,double &,double &);
cyclevarp nrcv(const int maxit,
         const double stepmax,const double b,int j,cyclevarp varxp,
         function<fd2(int,vec)> f)
{
    
    
    double fx,fy,fz,lower,upper,yy,zz;
    int i;
    cyclevarp varyp,varzp;
    vec y,z;
    zz=0.0;
// lower is lower bound for location of maximum.
    lower=-DBL_MAX;
// upper is upper bound for location of maximum.
    upper=DBL_MAX;
    
// Need value of f.value at x.
    fx=varxp.max;
// Current location.
    varzp=varxp;
    fz=varzp.max;
    z=varzp.locmax;
    

// Up to maxit iterations.
    for(i=0;i<maxit;i++)
    {
// The proposed step.  Keep within range.
        yy=newton(zz,varzp.der1,varzp.der2);

        yy=modit(zz,yy,stepmax,lower,upper);
// Get new function value, new derivative, and new second derivative.
        y=z;
        y(j)=z(j)+yy;
        varyp=cyclevarf(j,y,f);
        fy=varyp.max;
        
// Stop for 0 derivative.
        if(varyp.der1==0.0)return varyp;
        rebound(yy,varyp.der1,lower,upper);

// Look for adequate progress.
        
        
        if(yy!=0.0&&fy-fx>=b*yy*fabs(varyp.der1)) return varyp;
        
// Treat inadequate progress.
        
        
// See if sign of first derivative has changed.
        if(varyp.der1*varzp.der1<0.0)
        {
// Cubic interpolation.
            yy=davidon(zz,yy,fz,fy,varzp.der1,varyp.der1);
            y(j)=z(j)+yy;
            varyp=cyclevarf(j,y,f);
            
            if(varyp.der1==0.0)return varyp;
            rebound(yy,varyp.der1,lower,upper);

        }
//Update for next cycle.
       
        varzp=varyp;
        zz=yy;
        
    }
    return varyp;
}

