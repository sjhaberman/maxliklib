//Two-point algorithm for function maximization along a nonempty
//open line interval.
//Function and continuous gradient are f.value and f.grad.
//It is assumed that f.value approaches negative infinity
//if the argument approaches the upper or lower bounds of the line interval.
//It is also assumed that f.value has a unique critical point.
//Maximum number of iterations is maxit.
//The largest permitted step is stepmax>0.
//The improvement check is b>1.
#include<armadillo>
#include<float.h>
using namespace arma;
using namespace std;
double davidon(double,double,double,double,double,double);

struct fd1bv
{
    double value;
    vec grad;
    bool fin;
};
    
struct twopointgvarb
{
    vec locmax;
    double max;
    vec grad;
    bool fin;
};
twopointgvarb twopointgvarbf(vec x,function <fd1bv(vec)>f)
{
    fd1bv resultf;
    twopointgvarb result;
    resultf=f(x);
    result.locmax=x;
    result.fin=resultf.fin;
    if(result.fin)
    {
        result.max=resultf.value;
        result.grad=resultf.grad;
    }
    return result;
};
double davidon(double,double,double,double,double,double);
double modit(double,double,double,double,double);
void rebound(double,double,double &,double &);
twopointgvarb twopointbg(const int maxits,vec v,
                         twopointgvarb varx,const double stepmax,const double b,
                         function<fd1bv(vec)> f)
{
    double d,diru,dirx,diry,dirz,fu,fx,fy,fz,lower,upper,uu,xx,yy,zz;
    int i;
    twopointgvarb varu,vary,varz;
// x is oldest value.  y is next value.
    zz=0.0;
    yy=1.0;
// lower is lower bound for location of maximum.
    lower=-DBL_MAX;
// upper is upper bound for location of maximum.
    upper=DBL_MAX;
    
    
// Need fx for f.value(x) and dirx for d.
    if(!varx.fin)return varx;
    
    
    
// Current location.
    
    dirz=dot(v,varx.grad);
    if(dirz==0.0) return varx;
    fx=varx.max;
    varz=varx;
    fz=fx;
    vary=twopointgvarbf(varx.locmax+v,f);
    
    while(!vary.fin)
    {
        yy=0.5*yy;
        vary=twopointgvarbf(varx.locmax+yy*v,f);
    }
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
        varu=twopointgvarbf(varx.locmax+uu*v,f);
        while(!varu.fin)
        {
            uu=0.5*(uu+yy);
            varu=twopointgvarbf(varx.locmax+uu*v,f);
        }
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

