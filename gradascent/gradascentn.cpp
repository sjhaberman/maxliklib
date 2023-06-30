//Gradient ascent algorithm for function maximization of a twice continuously
//differentiable real function f.value of real vectors of dimension p on a
//nonempty open convex set O of p-dimensional vectors.  The gradient of f.value
//is f.grad.  The integer order must be 1 or 2.  If order is 2, then the
//Hessian f.hess of f.value is found even though it is not used.
//The strict pseudoconcavity condition described in the document
//convergence.pdf is assumed to apply for the real number a.  For the starting
//vector start, it is assumed that the value of f.value at start exceeds a.
//Parameters used are defined in mparams.
//The maximum number of main iterations is mparams.maxit.
//The maximum number of secondary iterations per main iteration
//is mparams.maxits. 
//The maximum fraction of a step toward a boundary is
//mparams.eta.
//For secondary iterations, the improvement check
//is mparams.gamma1<1.
//The cosine check  mparams.gamma2 is not used.
//The largest permitted step length is mparams.kappa>0.
//If a main iteration leads to a change of the function f less
//than mparams.tol, then iterations cease.
#include<armadillo>
using namespace std;
using namespace arma;
struct f2v
{
    double value;
    vec grad;
    mat hess;
};
struct maxf2v
{
    vec locmax;
    double max;
    vec grad;
    mat hess;
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
maxf2v gradascent(const int & order, const params & mparams, const vec & start,
    const function<f2v(const int &, const vec &)> f);;
f2v ngh(const int & , const double & , const vec & , 
    const function <f2v(const int & , const vec & )>);
maxf2v gradascentn(const int & order, const double & step, const params & mparams, const vec & start, const function<f2v(const int & , const vec &)> f)
{
     auto f2=[step,f](const int & order, const vec & x){return ngh(order,step,x,f);};    
     return gradascent(order, mparams, start, f2);
}
