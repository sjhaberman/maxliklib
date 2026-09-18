//Adjust distribution to fit mean of 0, variance of 1, skewness of 0,
//and fourth moment of 3.
#include<armadillo>
using namespace arma;
using namespace std;
struct pw{vec points; vec weights;};
struct params{bool print; uword maxit; uword maxits; double eta;
    double gamma1; double gamma2; double kappa; double tol;};
pw mdiapw(const int & , const params & , const char & ,
    const pw & , const mat & , const vec & );
pw mdia4(const int & order, const params & mparams,
    const char & algorithm, const pw & pws){
    double z;
    int i;
    mat T(pws.points.n_elem,4);
    for(i=0;i<pws.points.n_elem;i++){
        z=pws.points(i);
        T(i,0)=z;
        T(i,1)=z*T(i,0);
        T(i,2)=z*T(i,1);
        T(i,3)=z*T(i,2);
    }
    vec u={0.0,1.0,0.0,3.0};
    return mdiapw(order,mparams, algorithm, pws, T, u);
}
