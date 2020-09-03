//Compute Hermite polynomials up to order n for point x.
#include<armadillo>
using namespace std;
using namespace arma;
vec hermpoly(int & n,double & x)
{
    int i;
    double xn;
    vec h(n+1);
    h(0)=1.0;
    if(n>0)h(1)=x;
    if(n>1)
        for(i=1;i<n;i++)
        {
            xn=(double)i;
            h(i+1)=x*h(i)-xn*h(i-1);
        }
    return h;
}
            
            
            
