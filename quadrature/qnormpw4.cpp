//Compute points and weights for normal scores quadrature of given order
//with adjustment via MDIA for variance and kurtosis.
//Number of points is n.
#include<armadillo>
using namespace arma;
struct pw{vec points; vec weights;};
struct params{
    bool print;
    uword maxit;
    uword maxits;
    double eta;
    double gamma1;
    double gamma2;
    double kappa;
    double tol;
};
pw qnormpwe(const uword & n);
pw mdia4(const pw & , const params & );
pw qnormpw4(const int & n, const params & mparams){
    pw pws,pws4;
    pws=qnormpwe(n);
//Nothing to be done in this case.
    if(n<9)return pws;
    pws4=mdia4(pws,mparams);
    return pws4;
}
            
            
            
