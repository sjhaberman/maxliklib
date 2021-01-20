//Find maximum likelihood estimates for response model
//Responses are y.
//Predictors are x.
//Weights are w.
#include<armadillo>
using namespace arma;
using namespace std;
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
maxf2v nrv(const params & ,const vec & , function<f2v(vec &)>);
f2v genresplik(const std::vector<dat> & , const vec &);
maxf2v genrespmle(const params & mparams, const vector<dat> & data, const vec & start)
{
    maxf2v results;
    int p;
    p=start.n_elem;
    results.locmax.set_size(p);
    results.grad.set_size(p);
    results.hess.set_size(p,p);
    auto f=std::bind(genresplik,data,_1);
    results=nrv(mparams,start,f);
    return results;
}
