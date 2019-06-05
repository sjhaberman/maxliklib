//
//Find maximum likelihood estimates for logit model.
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
struct logitvar
{
    double maxloglik;
    vec mle;
    mat covmle;
};
struct fd2v
{
    double value;
    vec grad;
    mat hess;
};
struct regvars
{
    mat x;
    vec y;
    vec weight;
};
fd2v logitfunct(vec);
nrvvar nrv(int,int,double,vec,double,function<fd2v(vec)>);



logitvar logits(const int maxit,
                const int maxits,
                const double tol,
                const vec start,
                const double stepmax,
                const double b
                )
{
    nrvvar varx;
    logitvar results;
    varx=nrv(maxit,maxits,tol,start,b,logitfunct);
    results.maxloglik=varx.max;
    results.mle=varx.locmax;
    results.covmle=inv_sympd(-varx.hess);
    return results;
}
