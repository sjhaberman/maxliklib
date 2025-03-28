//Truncate iteration
//Old point is alpha0. New point is alpha1, stepmax is maximum step
//size. Lower bound is b.lower and upper bound is b.upper.
//eta<1 is used to limit changes.
#include<armadillo>
using namespace std;
using namespace arma;
struct bounds
{
     double lower;
     double upper;
};
double modit(const double & eta, const double & alpha0, const double & alpha1,
    const double & stepmax, const bounds & b)
{
    double result;
    if(alpha0<alpha1)
    {
        result=fmin(alpha0+stepmax,alpha1);
        if(isfinite(b.upper))result=fmin(result,alpha0+eta*(b.upper-alpha0));
    }
    else
    {
        result=fmax(alpha0-stepmax,alpha1);
        if(isfinite(b.lower))result=fmax(result,alpha0+eta*(b.lower-alpha0));
    }
    return result;
}
