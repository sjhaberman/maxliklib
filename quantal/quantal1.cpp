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
struct f1v
{
    double value;
    vec grad;
};
struct model
{
  char type;
  char transform;
};
f1v loglog1(ivec &,vec &);
f1v cumloglog1(ivec &,vec &);
f1v cumlogit1(ivec &,vec &);
f1v cumprobit1(ivec &,vec &);
f1v gradlogit1(ivec &,vec &);
f1v gradloglog1(ivec &,vec &);
f1v gradprobit1(ivec &,vec &);
f1v logit1(ivec &,vec &);
f1v logmean1(ivec &,vec &);
f1v multlogit1(ivec &,vec &);
f1v probit1(ivec &,vec &);
f1v ranklogit1(ivec &,vec &);
f1v quantal1(model & choice,ivec & y,vec & beta)
{
    switch(choice.type)
    {
        case 'C': switch(choice.transform)
        {
          case 'P': return cumprobit1(y,beta);
          case 'L': return cumlogit1(y,beta);
          default: return cumloglog1(y,beta);
        }
        case 'G': switch(choice.transform)
        {
          case 'P': return gradprobit1(y,beta);
          case 'L': return gradlogit1(y,beta);
          default: return gradloglog1(y,beta);
        }
        case 'M': return multlogit1(y,beta);
        case 'P': return logmean1(y,beta);
        case 'S': switch(choice.transform)
        {
          case 'P': return probit1(y,beta);
          case 'L': return logit1(y,beta);
          default: return loglog1(y,beta);
        }
        default: return ranklogit1(y,beta);
    }
}
