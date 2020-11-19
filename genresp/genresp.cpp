//Log likelihood component, gradient, and hessian matrix
//for response model with response y and parameter
//vector beta. The struct variable choice governs the model selection.
//choice.type='C' yields a cumulative model, choice.type='D'
//yields a continuous model, choice.type='G' yields
//a graded model, choice.type='M' yields a multinomial logit model,
//choice.type='P' yields a Poisson model with a logarithmic transformation,
//choice.type='R' yields a rank-logit model, and choice.type='S'
//yields a model for Bernoulli data.
//For cumulative, graded, and Bernoulli cases, choice.transform='G'
//yields a log-log transform, choice.transform='L' yields a logit
//transformation, and choice.transform='P' yields a probit transformation.
//For continuous models, choice.transform='G' yields a Gumbel model,
//choice.transform='L'  yields a logistic model, and choice.transform='N'
//yields a normal model.
#include<armadillo>
using namespace arma;
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
struct resp
{
  ivec iresp;
  vec dresp;
};
f2v cumloglog(ivec &,vec &);
f2v cumlogit(ivec &,vec &);
f2v cumprobit(ivec &,vec &);
f2v gradlogit(ivec &,vec &);
f2v gradloglog(ivec &,vec &);
f2v gradprobit(ivec &,vec &);
f2v gumbel(vec &,vec &);
f2v logistic(vec &,vec &);
f2v logit(ivec &,vec &);
f2v loglog(ivec &,vec &);
f2v logmean(ivec &,vec &);
f2v multlogit(ivec &,vec &);
f2v normal(vec &,vec &);
f2v normalv(vec &,vec &);
f2v probit(ivec &,vec &);
f2v ranklogit(ivec &,vec &);
f2v genresp(model & choice,resp & y,vec & beta)
{
    switch(choice.type)
    {
        case 'C': switch(choice.transform)
        {
          case 'G': return cumloglog(y.iresp,beta);
          case 'L': return cumlogit(y.iresp,beta);
          default: return cumprobit(y.iresp,beta);
        }
        case 'D':switch(choice.transform)
        {
          case 'G': return gumbel(y.dresp,beta);
          case 'L': return logistic(y.dresp,beta);
          default:
          if(y.dresp.n_elem>1)
          {
            return normalv(y.dresp,beta);
          }
          else
          {
            return normal(y.dresp,beta);
          }
        }
        case 'G': switch(choice.transform)
        {
          case 'G': return gradloglog(y.iresp,beta);
          case 'L': return gradlogit(y.iresp,beta);
          default: return gradprobit(y.iresp,beta);

        }
        case 'M': return multlogit(y.iresp,beta);
        case 'P': return logmean(y.iresp,beta);
        case 'R': return ranklogit(y.iresp,beta);
        default: switch(choice.transform)
        {
          case 'G': return loglog(y.iresp,beta);
          case 'L': return logit(y.iresp,beta);
          default: return probit(y.iresp,beta);
        }
    }
}
