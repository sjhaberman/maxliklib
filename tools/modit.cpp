//Truncate iteration
//Old point is x0. New point is x1, stepmax is maximum step
//size. Lower bound is lower and upper bound is upper.
#include<cmath>
double modit(double x0,double x1,double stepmax,double lower,double upper)
{
    double y;
    y=x1;
    y=fmin(x0+stepmax,y);
    y=fmin(upper,y);
    y=fmax(x0-stepmax,y);
    y=fmax(lower,y);
    
   
    return y;
    
}


