
//Find maximum likelihood estimates for multivariate normal response model.
//Maximum number of iterations is maxit.
//Tolerance is tol.
//Starting vector is start.
//Responses are global_y.
//Predictors are global_x.
//Offsets are global_offset.
//Weights are global_w.//Log likelihood and its gradient and Hessian
//for multivariate normal linear model with response vectors global_y and
//predictor array of matrices global_x.
//Weight vector global_w is used.

#include<armadillo>
using namespace arma;
using namespace std;
struct nrvvarb
{
    vec locmax;
    double max;
    vec grad;
    mat hess;
    bool fin;
};
struct mlevarb
{
    double maxloglik;
    vec mle;
    mat hess;
    bool fin;
};
struct fd2bv
{
    double value;
    vec grad;
    mat hess;
    bool fin;
};


fd2bv normalvlik(vec beta);
nrvvarb nrbv(int,int,double,vec,double,double,function<fd2bv(vec)>);

mlevarb normalvmle(const int maxit,
                const int maxits,
                const double tol,
                const vec start,
                const double stepmax,
                const double b
                )
{
    nrvvarb varx;
    mlevarb results;
    varx=nrbv(maxit,maxits,tol,start,stepmax,b,normalvlik);
    results.fin=varx.fin;
    if(varx.fin)
    {
        results.maxloglik=varx.max;
        results.mle=varx.locmax;
        results.hess=varx.hess;
    }
    return results;
}

