//Compute points and weights for Gauss-Hermite quadrature of given order.
//Order is n



#include<armadillo>
using namespace std;
using namespace arma;
vec hermcoeff(int);
double hermpoly(int,double);
struct pw
{
    vec points;
    vec weights;
};
pw hermpw(int n)
{
    double x,y;
    int i;
    vec coeff(n+1),p(n),w(n);
    cx_vec croots(n);
    pw pws;
    x=(double)n;
   
    coeff=hermcoeff(n);
    
    croots=roots(coeff);
   
    p=sort(real(croots));
    for(i=0;i<n;i++)
    {
        y=hermpoly(n-1,p(i));
        w(i)=exp(lgamma(x))/(x*y*y);
    }
    pws.points=p;
    pws.weights=w;
    return pws;
}
            
            
            
