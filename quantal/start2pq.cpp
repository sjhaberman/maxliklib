//Find starting values for maximum likelilood for 2-parameter model
//for binary or Poisson responses with a latent
//variable with a standard normal distribution and no covariates.

#include<armadillo>
using namespace arma;
extern char choice;
extern char choices[];
struct startval
{
    vec start;
    bool fin;
};
double invquantal(double);
double dquant(double,double);
using namespace arma;
startval start2pq(imat obs)
{
    
    
    double beta;
    int d,i,p;
    startval results;
    vec pr,b;
    mat ob,c,e;
    ob.set_size(obs.n_rows,obs.n_cols);
    ob=conv_to<mat>::from(obs);
    results.fin=true;
    p=ob.n_cols;
    
    d=p+p;
    results.start.set_size(d);
    pr.set_size(p);
    b.set_size(p);
    c.set_size(p,p);
    e.set_size(p,p);
    
    pr=trans(mean(ob));
    
    if(!all(pr)||(choice!='M'&&!all(pr<1.0)))
    {
        results.fin=false;
        return results;
        
    }
    c=cor(ob);
    
    eig_sym(b,e,c);
    e.col(p-1)=sqrt(b(p-1))*e.col(p-1);
    for(i=0;i<p;i++)
    {
        choice=choices[i];
        beta=invquantal(pr(i));
        
        results.start(i+i+1)=beta;
        results.start(i+i)=e(i,p-1)/dquant(pr(i),beta);
    }
    
    return results;
}

