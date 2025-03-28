//Value and first two derivatives of the logarithm
//of the standard logistic density at z.
//Value returned if order is 0, value and derivative returned if order is 1,
//and value, derivative, and second derivative returned if order is at least 2.
#include<armadillo>
using namespace arma;
struct f2
{
    double value;
    double der;
    double der2;
};
f2 logistic(const int & order, const double & z)
{
    f2 results;
    double w, x, y;
    y=exp(z);
    x=1+y;
    w=y/x;
    results.value=z-2.0*log(x);
    if(order>0) results.der=1-2.0*w;
    if(order>1) results.der2=-2.0*w*(1.0-w);
    return results;
}
