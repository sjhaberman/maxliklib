//Value and first two derivatives of the logarithm
//of the standard normal density at z.
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
f2 normal(const int & order, const double & z)
{
    f2 results;
    results.value=log_normpdf(z);
    if(order>0) results.der=-z;
    if(order>1) results.der2=-1.0;
    return results;
}
