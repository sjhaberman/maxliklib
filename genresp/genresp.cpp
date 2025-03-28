//Log likelihood component, gradient, and hessian matrix
//for response model with response y and parameter
//vector beta. The struct variable choice governs the model selection
//choice.type='B' yields a logit-beta model.
//choice.type='C' yields a cumulative model, choice.type='D'
//yields a continuous model, choice.type='E' yields a logit-dirichlet model,
//choice.type='G' yields
//a graded model, choice.type='H' yields a log-gamma model,
//choice.type='L' yields a multinomial logit model,
//choice.type='M' yields a model for the maximum of two independent Bernoulli
//random variables.  choice.type='N' yields a multivariate normal model,
//choice.type='P' yields a Poisson model with a logarithmic transformation,
//choice.type='R' yields a rank-logit model, choice.type='S'
//yields a model for Bernoulli data, and choice.type='T' yields a
//censored continuous model.
//For cumulative, graded, and Bernoulli cases, choice.transform='G'
//yields a complementary log-log transformation, choice.transform='H'
//yields a log-log transformation, choice.transform='L' yields a logit
//transformation, and choice.transform='N' yields a probit transformation.
//For continuous models, choice.transform='G' yields a minimum Gumbel model,
//choice.transform='H' yields a maximum Gumbel model,
//choice.transform='L'  yields a logistic model, and choice.transform='N'
//yields a normal model.  If order is 0, only the function is
//found, if order is 1, then the function and gradient are found.  If order is 2,
//then the function, gradient, and Hessian are returned.
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
f2v berresp(const int & , const char &, const resp &, const vec & );
f2v contresp(const int & , const char & , const resp & , const vec & );
f2v cumresp(const int & , const char &, const resp &, const vec &) ;
f2v gradresp(const int & , const char &, const resp &, const vec & );
f2v maxberresp(const int & , const char &, const resp &, const vec & );
f2v loggamma(const int & , const resp &, const vec & );
f2v logitbeta(const int & , const resp &, const vec & );
f2v logitdirichlet(const int & , const resp &, const vec & );
f2v logmean(const int & , const resp &, const vec & );
f2v maxberresp(const int & , const char & , const resp &, const vec & );
f2v multlogit(const int & , const resp &, const vec & );
f2v normalv(const int & , const resp &, const vec & );
f2v ranklogit(const int & , const resp &, const vec   & );
f2v truncresp(const int & , const char &, const resp & , const vec & );
f2v genresp(const int & order, const model & choice, const resp & y, const vec & beta)
{
    switch(choice.type)
    {
        case 'B': return logitbeta(order, y, beta); 
        case 'C': return cumresp(order, choice.transform, y, beta);
        case 'D': return contresp(order, choice.transform, y, beta);
        case 'E': return logitdirichlet(order, y, beta);
        case 'G': return gradresp(order, choice.transform, y, beta);
        case 'H': return loggamma(order, y, beta);
        case 'L': return multlogit(order, y, beta);
        case 'M': return maxberresp(order, choice.transform, y, beta);
        case 'N': return normalv(order, y, beta);
        case 'P': return logmean(order, y, beta);
        case 'R': return ranklogit(order, y, beta);
        case 'S': return berresp(order, choice.transform, y, beta);
        default: return truncresp(order, choice.transform, y, beta);
    }
}
