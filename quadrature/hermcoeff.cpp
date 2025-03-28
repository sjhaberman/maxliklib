//Compute coefficients for Hermite polynomial of given order.
//Order is n.
#include<armadillo>
using namespace arma;
vec hermcoeff(const int & n)
{
    double x;
    int i,k;    
    mat coeff(n+1,n+1,fill::zeros);
    coeff(0,0)=1.0;
    if(n>0)
    {
        for(i=1;i<=n;i++)
        {
            coeff(i,i)=1.0;
            coeff(i-1,i)=0.0;
        }
    }
    if(n>1)
    {
        for(i=1;i<n;i++)
        {
            x=double(i);
            coeff(0,i+1)=-x*coeff(0,i-1);
            for(k=1;k<i;k++)
            {
                coeff(k,i+1)=coeff(k-1,i)-x*coeff(k,i-1);
            }
        }
    }
    return reverse(coeff.col(n));
}
            
            
            
