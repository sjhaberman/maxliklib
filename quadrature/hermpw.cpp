//Compute points and weights for Gauss-Hermite quadrature of given order.
//Order is n.
#include<armadillo>
using namespace arma;
struct pw
{
    vec points;
    vec weights;
};
pw hermpw(const int & n)
{
    double x;
    int i;
    vec p(n),w(n);
    mat J(n,n,fill::zeros),K(n,n);
    pw pws;
    pws.points.set_size(n);
    pws.weights.set_size(n);
//Set up matrix for eigenvalue computation.
    if(n>1)
    {
        for(i=1;i<n;i++)
        {
            x=(double)i;
            J(i-1,i)=sqrt(x);
            J(i,i-1)=J(i-1,i);
        }
    }
//Eigenvalues and eigenvectors.
    eig_sym(p,K,J);
//Get weights.
    for(i=0;i<n;i++)
    {
        x=K(0,i);
        w(i)=x*x;
    }
    pws.points=p;
    pws.weights=w;
    return pws;
}
            
            
            
