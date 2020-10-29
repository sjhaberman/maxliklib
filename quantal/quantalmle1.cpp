//Find maximum likelihood estimates for quantal response model
//based on one parameter for a response.
//Maximum number of iterations is maxit.
//Tolerance is tol.
//Responses are y.
//Predictors are x.
//Weights are w.  Use conjugate gradient method.
#include<armadillo>
using namespace arma;
using namespace std;
struct f1v
{
    double value;
    vec grad;
};
struct maxf1v
{
    vec locmax;
    double max;
    vec grad;
};
struct paramga
{
    int maxit;
    int maxits;
    function<double(vec)> c;
    double eta;
    double gamma1;
    double gamma2;
    double kappa;
    double tol;
};
maxf1v conjgrad(const paramga&,const vec &,const function<f1v(vec &)>);
f1v quantallik1(vec &);
maxf1v quantalmle1(const paramga & gaparams,const vec & start)
{
    maxf1v results;
    int p;
    p=start.n_elem;
    results.locmax.set_size(p);
    results.grad.set_size(p);
    results=conjgrad(gaparams,start,quantallik1);
    return results;
}
