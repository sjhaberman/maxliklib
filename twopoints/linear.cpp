//Linearly interpolate to find 0 of function with
//value g0 at x0 and g1 at x1.


double linear(double x0,double x1,double g0,double g1)
{
    double diff;
    diff=x1-x0;
    return (x0-diff*g0/(g1-g0));
}
