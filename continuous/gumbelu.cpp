//Value and first two derivatives of the logarithm of the standard maximum Gumbel density at z.
//Value returned if order is 0, value and derivative returned if order is 1, and value,
//derivative, and second derivative returned if order is at least 2.
#include<armadillo>
using namespace arma;
struct f2
{
    double value;
    double der;
    double der2;
};
f2 gumbelu(const int & order, const double & z)
{
    f2 results;
    double y;
    y=exp(-z);
    results.value=-z-y;
    if(order>0) results.der=-1.0+y;
    if(order>1) results.der2=-y;
    return results;
}