//Find maximum of quadratic function with value f0 at x0, f1 at x1, and
//derivative g0 at x0.  Here f0 is not less than f1.
//Stepmax controls step size.
#include <cmath>
using namespace std;
double maxquad(double & x0,double & x1,double & f0,double & f1,
  double & g0,double & stepmax)
{
    double c,d,diff;
    diff=x1-x0;
    d=fmax(2.0*(g0-(f1-f0)/diff)/diff,fabs(g0)/stepmax);
    return x0+g0/d;
}
