//
//Find maximum likelihood estimates for quantal response model
//based on one parameter for a response.
//Maximum number of iterations is maxit.
//Tolerance is tol.
//Starting vector is start.
//Responses are y.
//Predictors are x.
#include<armadillo>
using namespace arma;
using namespace std;
struct nrvvar
{
    vec locmax;
    double max;
    vec grad;
    mat hess;
};
struct mlevar
{
    double maxloglik;
    vec mle;
    mat hess;
};
struct fd2v
{
    double value;
    vec grad;
    mat hess;
};


fd2v quantallik(vec);
nrvvar nrv(int,int,double,vec,double,double,function<fd2v(vec)>);



mlevar quantalmle(const int maxit,
                const int maxits,
                const double tol,
                const vec start,
                const double stepmax,
                const double b
                )
{
    nrvvar varx;
    mlevar results;
    varx=nrv(maxit,maxits,tol,start,stepmax,b,quantallik);
    results.maxloglik=varx.max;
    results.mle=varx.locmax;
    results.hess=varx.hess;
    return results;
}
