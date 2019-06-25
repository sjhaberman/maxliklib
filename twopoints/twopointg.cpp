//Two-point algorithm for function maximization along line.
//Function and continuous gradient are f.value and f.grad.
//It is assumed that f.value approaches negative infinity
//if the absolute value of the argument approaches infinity.
//It is also assumed that f.value has a unique critical point.
//Maximum number of iterations is maxit.
//If change in approximation to maximum is less
//than tol, then iterations cease.
//The largest permitted step is stepmax>0.
//The improvement check is b>1.
#include<armadillo>
#include<float.h>
using namespace arma;
using namespace std;
double davidon(double,double,double,double,double,double);

struct fd1v
{
    double value;
    double grad;
};
    
struct twopointgvar
{
    vec locmax;
    double max;
    vec grad;
};
twopointgvar twopointgvarf(vec,function <fd1v(vec)>);
double davidon(double,double,double,double,double,double);
double modit(double,double,double,double,double);
void rebound(double,double,double &,double &);
twopointgvar twopointg(const int maxits,vec v,
                         twopointgvar varx,const double stepmax,const double b,
                         function<fd1v(vec)> f)
{
    double d,diru,dirx,diry,dirz,fu,fx,fy,fz,lower,upper,uu,xx,yy,zz;
    int i;
    twopointgvar varu,vary,varz;
// x is oldest value.  y is next value.
    zz=0.0;
    yy=1.0;
// lower is lower bound for location of maximum.
    lower=0.0;
// upper is upper bound for location of maximum.
    upper=DBL_MAX;
    
// Need fx for f.value(x) and dirx for d.
    fx=varx.max;
// Current location.
    varz=varx;
    dirz=dot(v,varz.grad);
    
    vary=twopointgvarf(varx.locmax+v,f);
    fy=vary.max;
    diry=dot(v,vary.grad);
    if(diry<0.0)upper=yy;
    if(diry==0.0) return vary;

// Up to maxit iterations.
    for(i=0;i<maxits;i++)
    {
// The proposed step.
        uu=davidon(zz,yy,fz,fy,dirz,diry);
// Truncate if needed.
        uu=modit(yy,uu,stepmax,lower,upper);
        varu=twopointgvarf(varx.locmax+uu*v,f);
// Update limits.
        fu=varu.max;
        diru=dot(v,varu.grad);
        if(diru==0.0)return varu;
        if(fu-fx>=b*fabs(fu*uu))return varu;
        rebound(uu,diru,lower,upper);
        
        
// Prepare for next step.
        zz=yy;
        varz=vary;
        yy=uu;
        vary=varu;
        fz=fy;
        fy=fu;
        dirz=diry;
        diry=diru;

    }
    return vary;
}

