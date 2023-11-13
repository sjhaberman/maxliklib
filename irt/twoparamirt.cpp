//Maximum likelihood is applied to a two-parameter IRT model with a standard normal
//latent variable.
//The elementary case is considered in which all individuals have the same items and no data are
//missing.  In addition, all observations have unit weight.  Responses are all 0 or 1.
//They are obtained from a comma-separated file with no other entries.
//The model specifications are provided by the control file.
//The name of the file is read from standard input. 
//The file includes multiple lines of text.  Each line has two strings that are separated by
//a blank character. Each of these strings contains no blank character.
//The following pairs are used, with the first member of a pair a variable and the second
//the value of the variable.
//data infile (infile is the name of the data file).
//dist cdf, where cdf is N for normal, L for logistic, and G for Gumbel.  The default is L.
//method algorithm, where algorithm is G for gradient ascent, C for conjugate gradient ascent,
//N for modified Newton-Raphson, and L for Louis approximation.  The default is N.
//adapt aq, where aq is true for adaptive quadrature and false otherwise.
//quadrature quad, where quad is G for Gauss-Hermite and Q for normal quantiles.  The default is 
//G.  The last case is 
//points n, where n>1 is the number of quadrature points.  The default is 9.


#include<armadillo>
#include<string.h>
using namespace std;
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
//Select elements of vector.  all indicates all elements.  list lists elements.
struct xsel
{
  bool all;
  uvec list;
};
//Select elements of matrix.  all indicates all elements.  list lists elements in columns.
struct xselv
{
    bool all;
    umat list;
};
//Constant component of lambda.
struct lcomp
{
    int li;
    double value;
};
//Interaction of predictor and lambda.
struct lxcomp
{
    int li;
    int pi;
    double value;
};
//Interaction of predictor and lambda for predictor from another variable.
struct lxocomp
{
    int li;
    int pi;
    int ob;
    double value;
};
//Interaction of theta and lambda.
struct ltcomp
{
    int li;
    int th;
    double value;
};
//Interaction of beta and lambda.
struct lbcomp
{
    int li;
    int bi;
    double value;
};
//Interaction of beta and predictor with lambda.
struct lxbcomp
{
    int li;
    int pi;
    int bi;
    double value;
};
//Interaction of beta and predictor with lambda for predictor from another variable.
struct lxobcomp
{
    int li;
    int pi;
    int ob;
    int bi;
    double value;
};
//Interaction of beta and theta with lambda.
struct ltbcomp
{
    int li;
    int th;
    int bi;
    double value;
};
//Data structure.
struct dat
{
    resp y;
    vec x;
};    
//Specify a model.
//choice indicates transformation and type of model.
//dim is dimension of lambda;
//idim is dimension of integer response.
//ddim is dimension of double response.
//bdim is dimension of used beta elements.
//lcomps indicates constant components.
//lxcomps indicates components only dependent on the predictors.
//lxocomps indicates components only dependent on predictors from other variables.
//ltcomps indicates components only dependent on theta.
//lbcomps indicates components only dependent on parameters.
//lxbcomps indicates components dependent on predictors and parameters.
//lxobcomps indicates components dependent on predictors from other variables and parameters.
//ltbcompes indicates components dependent on theta and parameters.
//ithetas are integer elements from theta in response.
//dthetas are double elements from theta in response.
struct pattern
{
     model choice;
     int dim;
     int idim;
     int ddim;
     field<lcomp> lcomps;
     field<lxcomp>lxcomps;
     field<lxocomp>lxocomps;
     field<ltcomp>ltcomps;
     field<lbcomp>lbcomps;
     field<lxbcomp>lxbcomps;
     field<lxobcomp>lxobcomps;
     field<ltbcomp>ltbcomps;
     uvec ithetas;
     uvec dthetas;
};
//Basic quadrature rule.
struct pw
{
    vec points;
    vec weights;
};
// Weights and points for prior.
struct pwr
{
    double weight;
    double kernel;
    resp theta;
};
// Adaptive quadrature specifications.
// The choice to use is indicated by adapt, linselect shows the elements involved.
// quadselect shows the quadratic elements involved.
struct adq
{
    bool adapt;
    xsel linselect;
    xselv quadselect;
};
//Adaptive quadrature transformation.
struct dovecmat
{
    double s;
    vec v;
    mat m;
};
//Parameters for function maximization.
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
struct maxf2v
{
    vec locmax;
    double max;
    vec grad;
    mat hess;
};
maxf2v irtmle(const int & , const params & ,
    const char & ,const field<pattern> & ,
    const field<uvec> & , const uvec &  ,
    const field<field<uvec>> & , const uvec & ,
    const field<field <pwr>> & , const uvec & , 
    const field<adq> & , const uvec & ,
    field<dovecmat> & , 
    const field<field<dat>> & ,
    const field<field<xsel>> & , const uvec & , 
    const field<uvec> & , const uvec & ,
    const field<vec> & , const uvec & , const field<xsel> & , const uvec & ,
    const vec & , const xsel & , 
    const field<xsel> & , const uvec & , const vec &  );
pw hermpw(const int & );
pw qnormpw(const int & );
int main()
{
//d is model dimension, i and j are indices, m is number of observations, 
//n is number of quadrature points,  nr is number of observed responses per observation,
//nv is total number of responses.
//nc is number of control entries.
    bool aq=true;
    char algorithm='N',cdf='L',quad='G';
    int d, i, j, m, n=9, nc, nr, nv, order=2;
    params mparams;
    mparams.print=true;
    mparams.maxit=10;
    mparams.maxits=10;
    mparams.eta=0.5;
    mparams.gamma1=0.1;
    mparams.gamma2=0.1;
    mparams.kappa=3.0;
    mparams.tol=1.0;
    string controlfile,infile="infile.csv",key1,key2;
    imat responses;
    pw th;
    try{cin>>controlfile;}
    catch(...){cout<<"No control file specification"<<endl;return 1;}
    field<string>control;
    nc=0;
    try{control.load(controlfile);}catch(...){nc=-1;}
    if(nc>-1)nc=control.n_rows;
    if(nc>0){for(i=0;i<nc;i++)
    {
         key1=control(i,0);
         key2=control(i,1);
         if(key1.substr(0,2)=="ad")
         {
              if(key2[0]=='f') aq=false;
              continue;
         }
         if(key1.substr(0,2)=="da")
         {
              infile=key2;
              continue;
         }
         if(key1.substr(0,2)=="di")
         {
              if(key2[0]=='G') cdf='G';
              if(key2[0]=='N') cdf='N';
              continue;
         }
         
         if(key1.substr(0,2)=="me")
         {
              if(key2[0]=='G') {algorithm='G';order=1;}
              if(key2[0]=='C') {algorithm='C';order=1;}
              if(key2[0]=='L') algorithm='L';
              continue;
         }   
         if(key1.substr(0,2)=="po")
         {
              try{n=stoi(key2);}
              catch(...){n=9;}
              if(n<2) n=9;
              continue;
         }
         if(key1.substr(0,2)=="qu")
         {
              if(key2[0]=='Q')quad='Q';
              continue;
         }
         if(key1.substr(0,2)=="to")
         {
              try{mparams.tol=stod(key2);}
              catch(...){mparams.tol=1.0;}
              if(mparams.tol<=0.0) mparams.tol=1.0;
              continue;
         }
    }}
    responses.load(infile,csv_ascii);
    m=responses.n_rows;
    nr=responses.n_cols;
    nv=nr+1;
    d=2*nr;    
//Two patterns for observed and latent cases.
    field<pattern>  patterns(2);
//All observations satisfy the same model.
    field<uvec>  patternnumber(1);
    uvec   patno(m);
    field<field<uvec>>  selectobs(1);
    uvec selectobsno(m);
    field<field<pwr>>thetas(1);
    uvec  thetano(m); 
    field<adq>  scale(1);
    uvec  scaleno(m);
    field<dovecmat>  obsscale(m); 
    field<field<dat>>  data(m);
    field<field<xsel>>  selectbeta(1);
    uvec  selectbetano(m); 
    field<uvec>  betanumber(1);
    uvec  betanono(m);
    field<vec>  w(1);
    uvec  wno(m);
    field<xsel>  obssel(1);
    uvec obsselno(m);
    vec  obsweight(m);
    xsel  datasel; 
    field<xsel>  betasel(1);
    uvec  betaselno(m);
    vec   start(d);
    patterns(0).choice.type='S';
    patterns(0).choice.transform=cdf;
    patterns(0).dim=1;
    patterns(0).idim=1;
    patterns(0).ddim=0;
    patterns(0).lbcomps.set_size(1);
    patterns(0).lbcomps(0).li=0;
    patterns(0).lbcomps(0).bi=0;
    patterns(0).lbcomps(0).value=1.0;
    patterns(0).ltbcomps.set_size(1);
    patterns(0).ltbcomps(0).li=0;
    patterns(0).ltbcomps(0).bi=1;
    patterns(0).ltbcomps(0).th=0;
    patterns(0).ltbcomps(0).value=1.0;
    patterns(1).choice.type='D';
    patterns(1).choice.transform='N';
    patterns(1).dim=2;
    patterns(1).idim=0;
    patterns(1).ddim=1;
    patterns(1).dthetas.set_size(1);
    patterns(1).dthetas(0)=0;
    patterns(1).lcomps.set_size(1);
    patterns(1).lcomps(0).li=1;
    patterns(1).lcomps(0).value=1.0;
    patternnumber(0).set_size(nv);
    patternnumber(0).zeros();
    patternnumber(0)(nr)=1;
    patno.set_size(m);
    patno.zeros();
    selectobs(0).set_size(nv);
    selectobsno.set_size(m);
    selectobsno.zeros();
    thetas(0).set_size(n);
    th.points.set_size(n);
    th.weights.set_size(n);
//Set quadrature.
    if(quad=='G'){th=hermpw(n);}else{th=qnormpw(n);}
    for(i=0;i<n;i++)
    {
         thetas(0)(i).theta.dresp.set_size(1);
         thetas(0)(i).theta.iresp.set_size(0);
         thetas(0)(i).theta.dresp(0)=th.points(i);
         thetas(0)(i).weight=th.weights(i);
         thetas(0)(i).kernel=normpdf(th.points(i));
    }  
    thetano.zeros();
    scale(0).adapt=aq;
    if(aq)
    {
         scale(0).linselect.all=true;
         scale(0).quadselect.all=true;
    }
    scaleno.zeros();
    for(j=0;j<m;j++)
    {
        obsscale(j).v.set_size(1);
        obsscale(j).v(0)=0.0;
        obsscale(j).m.set_size(1,1);
        obsscale(j).m(0,0)=1.0;
        obsscale(j).s=1.0;
        data(j).set_size(nv);
        for(i=0;i<nr;i++)
        {
            
            data(j)(i).y.iresp.set_size(1);          
            data(j)(i).y.iresp(0)=responses(j,i);   
        }
        data(j)(nr).y.dresp.set_size(1);
    }
    betanumber(0).set_size(nv);
    selectbeta(0).set_size(nv);
    for(j=0;j<nr;j++)
    {
         selectbeta(0)(j).all=false;
         selectbeta(0)(j).list.set_size(2);
         selectbeta(0)(j).list(0)=2*j;
         selectbeta(0)(j).list(1)=2*j+1;
         betanumber(0)(j)=j;
    }
    selectbeta(0)(nr).all=false;
    betanumber(0)(nr)=nr;
    selectbetano.zeros();
    betanono.set_size(m);
    betanono.zeros();
    w(0).set_size(nv);
    w(0).ones();
    wno.zeros();
    obssel(0).all=true;
    obsselno.zeros();
    obsweight.ones();
    datasel.all=true;
    betasel(0).all=true;
    betaselno.zeros();
    start.zeros();
    for(j=0;j<nr;j++) start(j+j+1)=1.0;  
    maxf2v results;    
    results.grad.set_size(d);
    results.locmax.set_size(d);
    if(order>1)results.hess.set_size(d,d);  
    results=irtmle(order, mparams,
                     algorithm, 
                     patterns,
                     patternnumber, patno,
                     selectobs, selectobsno,
                     thetas, thetano, 
                     scale, scaleno,
                     obsscale, 
                     data,
                     selectbeta, selectbetano, 
                     betanumber, betanono,
                     w, wno, obssel, obsselno,
                     obsweight, datasel, 
                     betasel, betaselno, start);
    cout<<results.max<<endl;
    cout<<results.locmax<<endl;
    if(order>1)cout<<results.hess<<endl;   
    return 0;
}
