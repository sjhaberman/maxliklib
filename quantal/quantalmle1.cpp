
//Find maximum likelihood estimates for quantal response model
//based on one parameter for a response.
//Maximum number of iterations is maxit.
//Tolerance is tol.
//Starting vector is start.
//Responses are global_y.
//Predictors are global_x.
//Weights are global_w.
#include<armadillo>
using namespace arma;
using namespace std;
struct twopointgvar
{
    vec locmax;
    double max;
    vec grad;
    
};
struct mlevar1
{
    double maxloglik;
    vec mle;
    
};
struct fd1v
{
    double value;
    vec grad;
    
};


fd1v quantallik1(vec);
twopointgvar twopointgvarf(int,int,double,vec,double,double,function<fd1v(vec)>);



mlevar1 quantalmle1(const int maxit,
                const int maxits,
                const double tol,
                const vec start,
                const double stepmax,
                const double b
                )
{
    twopointgvar varx;
    mlevar1 results;
    varx=gradascent(maxit,maxits,tol,start,stepmax,b,quantallik1);
    results.maxloglik=varx.max;
    results.mle=varx.locmax;
    
    return results;
}
