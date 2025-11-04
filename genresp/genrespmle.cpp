//Find maximum likelihood estimates for response model.
//See genresplik.cpp and nrv.cpp for definitions of structs
//used.
//Newton-Raphson approach is used if algorithm is N.
//Louis approach is used if algorithm is L.
//Conjugate gradient method is used if algorithm is C.
//Gradient ascent is used if algorithm is G.
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
    uvec list;
};
struct pattern
{
    model choice;
    vec o;
    mat x;
};
maxf2v maxselect(const int &, const params & , const char & , const vec & ,
    const function<f2v(const int & , const vec & )> f);
f2v genresplik(const int & , const field<pattern> & ,
    const xsel & , const field<resp> & ,
    const field<xsel> & , const xsel & ,
    const vec & , const xsel & , const vec & );
maxf2v genrespmle(const int & order, const params & mparams,
    const char & algorithm, const field<pattern> & patterns,
    const xsel & patternnumber, const field<resp> & data,
    const field<xsel> & selectbeta, const xsel & selectbetano,
    const vec & w, const xsel & obssel, const vec & start)
{
    maxf2v results;
    int p;
    p=start.n_elem;
    results.locmax.set_size(p);
    results.grad.set_size(p);
    const function<f2v(const int & order,const vec & start)>f=
        [&patterns,&patternnumber,&data,
        &selectbeta,&selectbetano,
        &w,&obssel]
        (const int & order,const vec & start)
        {return genresplik(order, patterns,
        patternnumber, data,
        selectbeta, selectbetano,
        w, obssel, start);};
    results=maxselect(order, mparams, algorithm, start, f);
    return results;
}
