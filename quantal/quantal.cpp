//Log likelihood component, gradient, and hessian matrix
//for quantal model with response y and parameter
//vector beta. The struct variable choice governs the model selection.
//choice.type='C' yields a cumulative model, choice.type='G' yields
//a graded model, choice.type='M' yields a multinomial logit model,
//choice.type='P' yields a Poisson model with a logarithmic transformation,
//choice.type='R' yields a rank-logit model, and choice.type='S'
//yields a model for Bernoulli data.
//For cumulative, graded, and Bernoulli cases, choice.transform='G'
//yields a log-log transform, choice.transform='L' yields a logit
//transformation, and choice.transform='P' yields a probit transformation.
#include<armadillo>
using namespace arma;
using namespace std;
struct f2v
{
    double value;
    vec grad;
    mat hess;
};
struct model
{
  char type;
  char transform;
};
f2v loglog(ivec &,vec &);
f2v cumloglog(ivec &,vec &);
f2v cumlogit(ivec &,vec &);
f2v cumprobit(ivec &,vec &);
f2v gradlogit(ivec &,vec &);
f2v gradloglog(ivec &,vec &);
f2v gradprobit(ivec &,vec &);
f2v logit(ivec &,vec &);
f2v logmean(ivec &,vec &);
f2v multlogit(ivec &,vec &);
f2v probit(ivec &,vec &);
f2v ranklogit(ivec &,vec &);
f2v quantal(model & choice,ivec & y,vec & beta)
{
    switch(choice.type)
    {
        case 'C': switch(choice.transform)
        {
          case 'P': return cumprobit(y,beta);
          case 'L': return cumlogit(y,beta);
          default: return cumloglog(y,beta);
        }
        case 'G': switch(choice.transform)
        {
          case 'P': return gradprobit(y,beta);
          case 'L': return gradlogit(y,beta);
          default: return gradloglog(y,beta);
        }
        case 'M': return multlogit(y,beta);
        case 'P': return logmean(y,beta);
        case 'S': switch(choice.transform)
        {
          case 'P': return probit(y,beta);
          case 'L': return logit(y,beta);
          default: return loglog(y,beta);
        }
        default: return ranklogit(y,beta);
    }
}
