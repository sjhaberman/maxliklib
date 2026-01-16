//Adjust symmetric distribution to fit variance of 1 and fourth moment of 3.
#include<armadillo>
using namespace arma;
using namespace std;
//Function, gradient, and Hessian.
struct pw
{
    vec points;
    vec weights;
};
struct params
{
    bool print;
    int maxit;
    int maxits;
    double eta;
    double gamma1;
    double gamma2;
    double kappa;
    double tol;
};
pw mdiapw(const pw & , const mat & , const vec & , const params & );
vec hermpoly(const int & ,const double & );
pw mdia4(const pw & pws, const params & mparams)
{
    pw results;
    int i, k=4, order=2;
    vec f={1.0,1.0,1.0/sqrt(2.0),1.0/sqrt(6.0),1.0/sqrt(24.0)},h;
    mat T(pws.points.n_elem,2);
    for(i=0;i<pws.points.n_elem;i++){
        h=f%hermpoly(k,pws.points(i));
        T(i,0)=h(2);
        T(i,1)=h(4);
    }
    vec u(2);
    results=mdiapw(pws,T,u,mparams);
    return results;
}
