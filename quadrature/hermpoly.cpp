//Compute Hermite polynomials up to order n for point x.
#include<armadillo>
using namespace arma;
vec hermpoly(const uword & n, const double & x){
    uword i;
    double xn=0.0;
    vec h(n+1);
    h(0)=1.0;
    if(n>0)h(1)=x;
    for(i=1;i<n;i++){
        xn+=1.0;
        h(i+1)=x*h(i)-xn*h(i-1);
    }
    return h;
}
            
            
            
