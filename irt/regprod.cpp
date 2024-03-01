//Regression of sum of squares of y(i,j)-a(i)a(j) over selected pairs i and j.
//Algorithm uses minus the sum of squares and may 
//be gradient ascent (G), conjugate gradient (C), Newton-Raphson (N), or Lous (L). 


#include<armadillo>
using namespace std;
using namespace arma;
struct f2v
{
    double value;
    vec grad;
    mat hess;
};
struct resp
{
  ivec iresp;
  vec dresp;
};
//Parameters for function maximization.
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
struct maxf2v
{
    vec locmax;
    double max;
    vec grad;
    mat hess;
};
maxf2v maxselect(const int &, const params & , const char & , const vec & ,  
    const function<f2v(const int & , const vec & )> f);
f2v regprodf(const int & , const field<resp> & , const vec & );
maxf2v regprod(const int & order, const params & mparams, const char & algorithm,
    const field<resp> & y, const vec & start)
{
    maxf2v results;
    int p;
    p = start.n_elem;
    results.locmax.set_size(p);
    results.grad.set_size(p);
    if(order>1)results.hess.set_size(p,p);
    const function<f2v(const int & order, const vec & start)> f=
        [ &y](const int & order,const vec & start) mutable
             {return regprodf(order, y, start);};
    results=maxselect(order, mparams, algorithm, start, f);
    return results;
}

