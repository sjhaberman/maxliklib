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
    vec p,w;
    mat J(n,n,fill::zeros),K;
    pw pws;
//Set up matrix for eigenvalue computation.
    if(n>1){
        J.diag(1)=sqrt(regspace(1,n-1));
        J.diag(-1)=J.diag(1);
    }
//Eigenvalues and eigenvectors.
    eig_sym(p,K,J);
//Get weights.
    w=square(trans(K.row(0)));
    pws.points=p;
    pws.weights=w;
    return pws;
}

