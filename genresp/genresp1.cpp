//Log likelihood component and gradient
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
struct resp
{
  ivec iresp;
  vec dresp;
};
f1v cumloglog1(ivec &,vec &);
f1v cumlogit1(ivec &,vec &);
f1v cumprobit1(ivec &,vec &);
f1v gradlogit1(ivec &,vec &);
f1v gradloglog1(ivec &,vec &);
f1v gradprobit1(ivec &,vec &);
f1v gumbel1(vec &,vec &);
f1v logistic1(vec &,vec &);
f1v logit1(ivec &,vec &);
f1v loglog1(ivec &,vec &);
f1v logmean1(ivec &,vec &);
f1v multlogit1(ivec &,vec &);
f1v normal1(vec &,vec &);
f1v normalv1(vec &,vec &);
f1v probit1(ivec &,vec &);
f1v ranklogit1(ivec &,vec &);
f1v genresp1(model & choice,resp & y,vec & beta)
{
    switch(choice.type)
    {
        case 'C': switch(choice.transform)
        {
          case 'G': return cumloglog1(y.iresp,beta);
          case 'L': return cumlogit1(y.iresp,beta);
          default: return cumprobit1(y.iresp,beta);
        }
        case 'D':switch(choice.transform)
        {
          case 'G': return gumbel1(y.dresp,beta);
          case 'L': return logistic1(y.dresp,beta);
          default:
          if(y.dresp.n_elem>1)
          {
            return normalv1(y.dresp,beta);
          }
          else
          {
            return normal1(y.dresp,beta);
          }
        }
        case 'G': switch(choice.transform)
        {
          case 'G': return gradloglog1(y.iresp,beta);
          case 'L': return gradlogit1(y.iresp,beta);
          default: return gradprobit1(y.iresp,beta);

        }
        case 'M': return multlogit1(y.iresp,beta);
        case 'P': return logmean1(y.iresp,beta);
        case 'R': return ranklogit1(y.iresp,beta);
        default: switch(choice.transform)
        {
          case 'G': return loglog1(y.iresp,beta);
          case 'L': return logit1(y.iresp,beta);
          default: return probit1(y.iresp,beta);
        }
    }
}
