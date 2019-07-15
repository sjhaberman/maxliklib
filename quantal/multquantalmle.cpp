//Log likelihood and its gradient and Hessian
//for multinomial quantal model with integerresponse vector global_y and
//predictor array of matrices global_x.
//Weight vector global_w is used.

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



fd2v multquantallik(vec);
nrvvar nrv(const int,const int,const double,vec,const double,const double b,function<fd2v(vec)>);
mlevar multquantalmle(const int maxit,
                      const int maxits,
                      const double tol,
                      const vec start,
                      const double stepmax,
                      const double b)
{
    nrvvar varx;
    mlevar results;
    varx=nrv(maxit,maxits,tol,start,stepmax,b,multquantallik);
    results.maxloglik=varx.max;
    results.mle=varx.locmax;
    results.hess=varx.hess;
    return results;
    
}
