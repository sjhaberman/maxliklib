//Interpolate cubic function to fit function f0 with derivative g0 at x0
//and function f1 with derivative g1 at x1
//
#include<cmath>
double linear(double,double,double,double);
double davidon(double x0,double x1,double f0,double f1,double g0,double g1)
{
    double c,d,diff,diff2,newx,s;
    diff=x1-x0;
    if(diff==0.0)return x1;
    diff2=diff*diff;
    c=(3.0*(f1-f0)-(2.0*g0+g1)*diff)/diff2;
    d=((g0+g1)*diff-2.0*(f1-f0))/(diff2*diff);
    s=c*c-3.0*d*g0;
    if(c<0.0&&s>0.0)
    {
// Cubic interpolation plausible.
       newx= x0-g0/(c-sqrt(s));
    }
    else
    {
// Just use linear interpolation if possible.
        if(g1-g0!=0.0)
        {
            newx=linear(x0,x1,g0,g1);
        }
        else
// If all else fails, use the midpoint.
        {
            newx=(x0+x1)/2.0;
            
        }
    }
    return newx;
}
