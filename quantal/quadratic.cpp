//Quadratically interpolate to estimate maximum location
//of function with
//value f0 at x0,  f1 at x1, and f2 at x2.


double quadratic(double x0,double x1,double x2,double f0,double f1,double f2)
{
    double denom,l01,l02,l12,newx,numer;
    
    l01=(x2-x0)*(x2-x1);
    l02=(x1-x0)*(x1-x2);
    l12=(x0-x1)*(x0-x2);
    denom=2.0*(f0/l12+f1/l02+f2/l01);
    numer=(x1+x2)*f0/l12+(x0+x2)*f1/l02+(x0+x1)*f2/l01;
//Interpolating functions is strictly concave.
    if(denom<0.0)
    {
        newx=numer/denom;
    }
    else
    {
// try midpoint.
        newx=(x0+x1+x2)/3.0;
    }
    return newx;
}
