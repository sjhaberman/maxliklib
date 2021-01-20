//Log likelihood component and gradient
//for response model with response y and parameter
//vector beta. The struct variable choice governs the model selection.
//choice.type='C' yields a cumulative model, choice.type='D'
//yields a continuous model, choice.type='G' yields
//a graded model, choice.type='L' yields a multinomial logit model,
//choice.type='M' yields a model for the maximum of two independent Bernoulli
//random variables.
//choice.type='P' yields a Poisson model with a logarithmic transformation,
//choice.type='R' yields a rank-logit model, choice.type='S'
//yields a model for Bernoulli data, and choice.type='T' yields a
//censored continuous model.
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
f1v berresp1(const char &, const ivec &, const vec & );
f1v contresp1(const char & , const vec & , const vec & );
f1v cumresp1(const char &, const ivec &, const vec &) ;
f1v gradresp1(const char &, const ivec &, const vec & );
f1v maxberresp1(const char &, const ivec &, const vec & );
f1v logmean1(const ivec &, const vec & );
f1v maxberresp1(const char & , const ivec &, const vec & );
f1v multlogit1(const ivec &, const vec & );
f1v ranklogit1(const ivec &, const vec & );
f1v truncresp1(const char &, const resp & , const vec & );
f1v genresp1(const model & choice, const resp & y, const vec & beta)
{
    switch(choice.type)
    {  
        case 'C': return cumresp1(choice.transform,y.iresp,beta);
        case 'D': return contresp1(choice.transform,y.dresp,beta);
        case 'G': return gradresp1(choice.transform,y.iresp,beta); 
        case 'L': return multlogit1(y.iresp,beta);
        case 'M': return maxberresp1(choice.transform,y.iresp,beta);
        case 'P': return logmean1(y.iresp,beta);
        case 'R': return ranklogit1(y.iresp,beta);
        case 'S': return berresp1(choice.transform,y.iresp,beta);
        default: return truncresp1(choice.transform,y,beta);
    }
}
