//Find maximum likelihood estimates for response model.
//See genresplik.cpp and nrv.cpp for definitions of structs
//used.
//Newton-Raphson approach is used if algorithm is N.
//Louis approach is used if algorithm is L.
//Conjugate gradient method is used if algorithm is C.
//Gradient ascent is used if algorithm is G.
#include<armadillo>
using namespace arma;
using namespace std::placeholders;
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
    int maxit;
    int maxits;
    double eta;
    double gamma1;
    double gamma2;
    double kappa;
    double tol;
};
struct model
{
  char type;
  char transform;
};
struct resp
{
  ivec iresp;
  vec dresp;
};
struct xsel
{
  bool all;
  ivec list;
};
struct dat
{
     model choice;
     double weight;
     resp dep;
     vec offset;
     mat indep;
     xsel xselect;
};
maxf2v conjgrad(const int &, const params & , const vec & ,
           const std::function<f2v(const int & , const vec & )> f);
maxf2v gradascent(const int &, const params & , const vec & ,
           const std::function<f2v(const int & , const vec & )> f);
maxf2v nrv(const int &, const params & , const vec & ,
           const std::function<f2v(const int & , const vec & )> f);

f2v genresplik(const int & , const std::vector<dat> & , const vec &);
maxf2v genrespmle(const int & order, const params & mparams,
    const char & algorithm, const std::vector<dat> & data, const vec & start)
{
    maxf2v results;
    int p;
    p=start.n_elem;
    results.locmax.set_size(p);
    results.grad.set_size(p);
    if(algorithm=='N'||algorithm=='L')results.hess.set_size(p,p);
    auto f=std::bind(genresplik, _1, data, _2);
    if(algorithm=='N'||algorithm=='L')results=nrv(order, mparams, start, f);
    if(algorithm=='C')results=conjgrad(order, mparams, start, f);
    if(algorithm=='G')results=gradascent(order, mparams, start, f);
    return results;
}
