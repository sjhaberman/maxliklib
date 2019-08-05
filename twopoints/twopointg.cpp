//Two-point algorithm for function maximization along line.
//Function and continuous gradient are f.value and f.grad.
//It is assumed that f.value approaches negative infinity
//if the absolute value of the argument approaches infinity.
//It is also assumed that f.value has a unique critical point.
//Maximum number of iterations is maxit.
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
    vec grad;
};
    
struct twopointgvar
{
    vec locmax;
    double max;
    vec grad;
};
twopointgvar twopointgvarf(vec x,function<fd1v(vec)> f)
{
    fd1v resultf;
    twopointgvar result;
    resultf=f(x);
    result.locmax=x;
    result.max=resultf.value;
    result.grad=resultf.grad;
    return result;
};
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
// See if nothing to be done.
    dirz=dot(v,varx.grad);
    if(dirz==0.0) return varx;
// x is oldest value.  y is next value.
    zz=0.0;
    yy=1.0;
// lower is lower bound for location of maximum.
    lower=-DBL_MAX;
// upper is upper bound for location of maximum.
    upper=DBL_MAX;
    
// Need fx for f.value(x) and dirx for d.
    fx=varx.max;
// Current location.
    varz=varx;
   
    fz=fx;
    
    vary=twopointgvarf(varx.locmax+v,f);
    fy=vary.max;
    diry=dot(v,vary.grad);
    
    if(diry==0.0) return vary;
    
    rebound(zz,dirz,lower,upper);
    rebound(yy,diry,lower,upper);
//Switch.
    
    if(fz>fy)
    {
        uu=yy;
        yy=zz;
        zz=uu;
        fu=fy;
        fy=fz;
        fz=fu;
        varu=vary;
        vary=varz;
        varz=varu;
        diru=diry;
        diry=dirz;
        dirz=diru;
    }
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
        if(fu-fx>=b*fabs(diru*uu))return varu;
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
// Switch order.
        if(fz>fy)
        {
            uu=yy;
            yy=zz;
            zz=uu;
            fu=fy;
            fy=fz;
            fz=fu;
            varu=vary;
            vary=varz;
            varz=varu;
            diru=diry;
            diry=dirz;
            dirz=diru;
        }
    }
    return vary;
}

