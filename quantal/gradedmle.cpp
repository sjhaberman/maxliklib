//
//Find maximum likelihood estimates for graded response model
//Maximum number of iterations is maxit.
//Tolerance is tol.
//Starting vector is start.
//Responses are global_y.
//Predictors are global_x.
//Offsets are global_offset.
//Weights are global_w.
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


fd2bv gradedlik(vec);
nrvvarb nrbv(int,int,double,vec,double,double,function<fd2bv(vec)>);



mlevarb gradedmle(const int maxit,
                const int maxits,
                const double tol,
                const vec start,
                const double stepmax,
                const double b
                )
{
    
    nrvvarb varx;
    mlevarb results;
    varx=nrbv(maxit,maxits,tol,start,stepmax,b,gradedlik);
    
    results.fin=varx.fin;
    if(varx.fin)
    {
        results.maxloglik=varx.max;
        results.mle=varx.locmax;
        results.hess=varx.hess;
    }
    return results;
}
