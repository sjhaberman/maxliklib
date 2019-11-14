
//Find maximum likelihood estimates for multivariate normal response model.
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
struct twopointgvarb
{
    vec locmax;
    double max;
    vec grad;
    
    bool fin;
};
struct mlevar1b
{
    double maxloglik;
    vec mle;
   
    bool fin;
};
struct fd1bv
{
    double value;
    vec grad;
    
    bool fin;
};


fd1bv normalvlik1(vec beta);
twopointgvarb gradascentb(int,int,double,vec,double,double,function<fd1bv(vec)>);

mlevar1b normalvmle1(const int maxit,
                const int maxits,
                const double tol,
                const vec start,
                const double stepmax,
                const double b
                )
{
    twopointgvarb varx;
    mlevar1b results;
    varx=gradascentb(maxit,maxits,tol,start,stepmax,b,normalvlik1);
    results.fin=varx.fin;
    if(varx.fin)
    {
        results.maxloglik=varx.max;
        results.mle=varx.locmax;
        
    }
    return results;
}

