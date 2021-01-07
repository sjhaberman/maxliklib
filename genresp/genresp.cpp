//Log likelihood component, gradient, and hessian matrix
//for response model with response y and parameter
//vector beta. The struct variable choice governs the model selection
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
f2v berresp(const char &, const ivec &, const vec & );
f2v contresp(const char & , const vec & , const vec & );
f2v cumresp(const char &, const ivec &, const vec &) ;
f2v gradresp(const char &, const ivec &, const vec & );                                                              
f2v maxberresp(const char &, const ivec &, const vec & );
f2v logmean(const ivec &, const vec & );
f2v maxberresp(const char & , const ivec &, const vec & );
f2v multlogit(const ivec &, const vec & );
f2v ranklogit(const ivec &, const vec & );
f2v truncresp(const char &, const resp & , const vec & );
f2v genresp(const model & choice, const resp & y, const vec & beta)
{
    switch(choice.type)
    {  
        case 'C': return cumresp(choice.transform,y.iresp,beta);
        case 'D': return contresp(choice.transform,y.dresp,beta);
        case 'G': return gradresp(choice.transform,y.iresp,beta); 
        case 'L': return multlogit(y.iresp,beta);
        case 'M': return maxberresp(choice.transform,y.iresp,beta);
        case 'P': return logmean(y.iresp,beta);
        case 'R': return ranklogit(y.iresp,beta);
        case 'S': return berresp(choice.transform,y.iresp,beta);
        default: return truncresp(choice.transform,y,beta);
    }
}
