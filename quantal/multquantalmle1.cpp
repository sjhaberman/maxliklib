//Maximum likelihood and location of maximum likelihood estimate
//for multinomial quantal model with integer response vector global_y and
//predictor array of matrices global_x.
//Weight vector global_w is used.
//Maximum number of iterations is maxit.
//Maximum for secondary iterations is maxits.
//Tolerance is tol.
//Starting vector is start.
//Maximum step is stepmax.
//Progress constant is b.



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



fd1v multquantallik1(vec);
twopointgvar twopointgvarf(vec,function<fd1v(vec)>);
twopointgvar gradascent(const int,const int,const double,const vec,
    const double,const double,function<fd1v(vec)> f);

mlevar1 multquantalmle1(const int maxit,
                      const int maxits,
                      const double tol,
                      const vec start,
                      const double stepmax,
                      const double b)
{
    twopointgvar varx;
    mlevar1 results;
    varx=gradascent(maxit,maxits,tol,start,stepmax,b,multquantallik1);
    results.maxloglik=varx.max;
    results.mle=varx.locmax;
    
    return results;
    
}
