//Maximum likelihood is applied to a generalized partial credit IRT model
//with a standard normal latent variable.
//The elementary case is considered in which all individuals have the same items and no data are
//missing.  In addition, all observations have unit weight.  Responses
//are obtained from a file infile for a matrix of integers.
//The model specifications are provided by the control file.
//The name of the file is read from standard input. 
//The file includes multiple lines of text.  Each line has two strings that are separated by
//a blank character. Each of these strings contains no blank character.
//The following pairs are used, with the first member of a pair a variable and the second
//the value of the variable.
//data infile (infile is the name of the data file).
//sf startfile is the file containing the vector of starting values.  (If this pair is not
//given, then the default procedure is used.)
//outfile, where outfile is the name of the output file.
//fflag is false if no output file is used.
//pflag is false if nothing is printed in ascii form.
//method algorithm, where algorithm is G for gradient ascent, C for conjugate gradient ascent,
//N for modified Newton-Raphson, and L for Louis approximation.  The default is N.
//tol indicates the convergence tolerance.  The default is 0.001.
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
//Specify a model.
//choice is model distribution.
//o is constant vector.
//x is tranformation from beta elements used to lambda that does not involve theta.
//c is transformation from beta elements used and theta double elements  used to lambda.
struct pattern
{
     model choice;
     vec o;
     mat x;
     cube c;
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
    const char & , const field<pattern> & , 
    const field<xsel> & , const xsel &  ,
    const field<field<resp>> & , const field<field<pwr>> & , const xsel & ,
    const field<adq> & , const xsel & , field<dovecmat> & ,
    const field<xsel> & , const field<xsel> & , const xsel & ,
    const field<xsel> & , const field<xsel> & , const xsel & ,
    const field<xsel> & , const field<xsel> & , const xsel & ,
    const field<xsel> & , const field<xsel> & , const xsel & ,
    const field<xsel> & , const field<xsel> & , const xsel & ,
    const field<vec> & , const xsel & , const field<xsel> & , const xsel & ,
    const vec & , const xsel & , 
    const field<xsel> & , const xsel & , const vec &  );
void savmaxf2v(const int & , const maxf2v & , const string & , const bool & , const bool & );
vec startgpcm(const int & , const params & , const char & , const imat & );
pw hermpw(const int & );
pw qnormpwe(const int & );
int main()
{
//d is model dimension, h, i, j, and k are indices, m is number of observations, 
//n is number of quadrature points,  nr is number of observed responses per observation,
//nmaxmin is smallest maximum value of a response variable,
//nmaxmax is largest maximum value of a responses variable.
//nmaxrange is range of maximum values of response variables.
//nv is total number of responses.
//nc is number of control entries.
    bool aq=true,readflag,fflag=true,pflag=true,sflag=true;
    char algorithm='N',cdf='L',quad='G';
    int d, h, i, j, k, m, n=9, nc, nr, nmaxmin, nmaxmax,nmaxrange, nv, order=2;
    params mparams;
    mparams.print=true;
    mparams.maxit=100;
    mparams.maxits=10;
    mparams.eta=0.5;
    mparams.gamma1=0.1;
    mparams.gamma2=0.1;
    mparams.kappa=3.0;
    mparams.tol=0.001;
    string controlfile,infile="infile.csv", outfile="outfile", sfile, key1, key2;
    imat responses;
    pw th;
    try{cin>>controlfile;}
    catch(...){cout<<"Name of control file not read."<<endl; return 1;}
    field<string>control;
    nc=0;
    try{control.load(controlfile);}
    catch(...){cout<<"Control file not read."<<endl; return 1;}
    nc=control.n_rows;
    if(nc>0){for(i=0;i<nc;i++)
    {
         key1=control(i,0);
         key2=control(i,1);
         if(key1.substr(0,2)=="ad")
         {
              if(key2[0]=='f') aq=false;
              continue;
         }
         if(key1.substr(0,2)=="sf")
         {
              sfile=key2;
              sflag=false;
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
              catch(...){mparams.tol=0.001;}
              if(mparams.tol<=0.0) mparams.tol=0.001;
              continue;
         }
         if(key1.substr(0,2)=="ou")
         {
              outfile=key2;
              continue;
         }
         if(key1.substr(0,2)=="ff")
         {
              if(key2[0]=='f') fflag=false;
              continue;
         }
         if(key1.substr(0,2)=="pf")
         {
              if(key2[0]=='f') pflag=false;
              continue;
         }
    }}
    try
    {
         responses.load(infile);
    }
    catch(...)
    {
         cout<<"Responses not loaded"<<endl; return 1;
    }
    m=responses.n_rows;
    nr=responses.n_cols;
    irowvec nmax(nr);
    nmax=max(responses,0);
    nv=nr+1;
    d=nr+sum(nmax);
    vec start(d);
    maxf2v results;    
    results.grad.set_size(d);
    results.locmax.set_size(d);
    if(order>1)results.hess.set_size(d,d);
    nmaxmax=max(nmax);
    nmaxmin=min(nmax);
    nmaxrange=nmaxmax-nmaxmin;
//Table of values of maxima of response variables and mapping of response maxima to
//patterns.
    uvec nmaxtab(nmaxrange+1,fill::zeros),nmaxmap(nmaxrange+1,fill::zeros);
//Classify by number of categories.
    for(i=0;i<nr;i++)
    {
         j=nmax(i)-nmaxmin;
         nmaxtab(j)=nmaxtab(j)+1;
    }
    j=0;
    for(i=0;i<=nmaxrange;i++)
    {
        nmaxmap(i)=j;
        if(nmaxtab(i)>0) j=j+1;
    }
    uvec nmaxes(j);
    k=0;
    for(i=0;i<=nmaxrange;i++)
    {
        if(nmaxtab(i)>0)
        {
            nmaxes(k)=nmaxmin+i;
            k=k+1;
        }
    } 
//j+1 patterns for observed and latent cases.
    field<pattern>  patterns(j+1);
    field<xsel>  patternnumber(1);
    xsel   patno;
    field<field<resp>>  data(m);
    field<field <pwr>>  thetas(n);
    xsel  thetano;
    field<adq>  scale(1);
    xsel  scaleno;
    field<dovecmat>  obsscale(m);  
    field<xsel>  selectbeta(nv);
    field<xsel> selectbetano(1);
    xsel selbetano;
    field<xsel>  selectbetac(j+1);
    field<xsel> selectbetacno(1);
    xsel selbetacno;
    field<xsel>  selectthetai(1);
    field<xsel> selectthetaino(1);
    xsel selthetaino;
    field<xsel>  selectthetad(2);
    field<xsel> selectthetadno(1);
    xsel selthetadno;
    field<xsel>  selectthetac(1);
    field<xsel> selectthetacno(1);
    xsel selthetacno;
    field<vec>  w(1);
    xsel  wno;
    field<xsel>  obssel(1);
    xsel obsselno;
    vec  obsweight(m);
    xsel  datasel; 
    field<xsel>  betasel(1);
    xsel  betaselno;
    for(i=0;i<j;i++)
    {
         k=nmaxes(i);
         patterns(i).choice.type='L';
         patterns(i).o.set_size(k);
         patterns(i).o.zeros();
         patterns(i).x.set_size(k,k+1);
         patterns(i).x.cols(0,k-1)=eye(k,k);
         patterns(i).x.col(k).zeros();
         patterns(i).c.set_size(k,1,1);
         for(h=0;h<k;h++) patterns(i).c(h,0,0)=(double) (h+1); 
    }
    patterns(j).choice.type='D';
    patterns(j).choice.transform='N';
    patterns(j).o.set_size(2);
    patterns(j).o(0)=0.0;
    patterns(j).o(1)=1.0;
    patternnumber(0).all=false;
    patternnumber(0).list.set_size(nv);
    for(i=0;i<nr;i++)patternnumber(0).list(i)=nmaxmap(nmax(i)-nmaxmin);
    patternnumber(0).list(nr)=j;
    patno.all=false;
    patno.list.set_size(m);
    patno.list.zeros();
    pw pws;
    thetas(0).set_size(n);
    pws.points.set_size(n);
    pws.weights.set_size(n);
//Set quadrature.
    if(quad=='G'){pws=hermpw(n);}else{pws=qnormpwe(n);} 
    for(i=0;i<n;i++)
    {
         thetas(0)(i).theta.dresp.set_size(1);
         thetas(0)(i).theta.iresp.set_size(0);
         thetas(0)(i).theta.dresp(0)=pws.points(i);
         thetas(0)(i).weight=pws.weights(i);
         thetas(0)(i).kernel=normpdf(pws.points(i));
    }
    thetano.all=false;
    thetano.list.set_size(m);  
    thetano.list.zeros();
    scale(0).adapt=aq;
    if(aq)
    {
         scale(0).linselect.all=true;
         scale(0).quadselect.all=true;
    }
    scaleno.all=false;
    scaleno.list.set_size(m);
    scaleno.list.zeros();
    for(h=0;h<m;h++)
    {
        obsscale(h).v.set_size(1);
        obsscale(h).v(0)=0.0;
        obsscale(h).m.set_size(1,1);
        obsscale(h).m(0,0)=1.0;
        obsscale(h).s=1.0;
        data(h).set_size(nv);
        for(i=0;i<nr;i++)
        {
            
            data(h)(i).iresp.set_size(1);          
            data(h)(i).iresp(0)=responses(h,i);   
        }
        data(h)(nr).dresp.set_size(1);
    }
    k=0;
    for(i=0;i<nr;i++)
    {
         selectbeta(i).all=false;
         selectbeta(i).list.set_size(nmax(i)+1);
         selectbeta(i).list=regspace<uvec>(k,k+nmax(i));
         k=k+nmax(i)+1;
    }
    selectbeta(nr).all=false;
    for(i=0;i<j;i++)
    {
         selectbetac(i).all=false;
         selectbetac(i).list.set_size(1);
         selectbetac(i).list(0)=nmaxes(i);
    }
    selectbetac(j).all=false;
    selectbetano(0).all=false;
    selectbetano(0).list.set_size(nv);
    selectbetano(0).list=regspace<uvec>(0,nr);
    selectbetacno(0).all=false;
    selectbetacno(0).list.set_size(nv);
    selectbetacno(0).list=patternnumber(0).list;
    selbetano.all=false;
    selbetano.list.set_size(m);
    selbetano.list.zeros();
    selbetacno.all=false;
    selbetacno.list.set_size(m);
    selbetacno.list.zeros();
    selectthetai.set_size(1);
    selectthetai(0).all=false;
    selectthetaino(0).list.set_size(m);
    selectthetaino(0).list.zeros();
    selthetaino.all=false;
    selthetaino.list.set_size(m);
    selthetaino.list.zeros();
    selectthetad.set_size(2);
    selectthetad(0).all=false;
    selectthetad(1).all=true;
    selectthetadno(0).all=false;
    selectthetadno(0).list.set_size(nv);
    selectthetadno(0).list.zeros();
    selectthetadno(0).list(nr)=1;
    selthetadno.all=false;
    selthetadno.list.set_size(m);
    selthetadno.list.zeros();
    selectthetac.set_size(2);
    selectthetac(0).all=true;
    selectthetac(1).all=false;
    selectthetacno(0).all=false;
    selectthetacno(0).list.set_size(nv);
    selectthetacno(0).list.zeros();
    selectthetacno(0).list(nr)=1;
    selthetacno.all=false;
    selthetacno.list.set_size(m);
    selthetacno.list.zeros();
    w(0).set_size(m);
    w(0).ones();
    wno.all=false;
    wno.list.set_size(m);
    wno.list.zeros();
    obssel(0).all=true;
    obsselno.all=false;
    obsselno.list.set_size(m);
    obsselno.list.zeros();
    obsweight.ones();
    datasel.all=true;
    betasel(0).all=true;
    betaselno.all=false;
    betaselno.list.set_size(m);
    betaselno.list.zeros();
    if(sflag)
    {
        start=startgpcm(order,mparams,algorithm,responses);
        if(start.has_nan())
        {
            cout<<"MLE does not exist"<<endl; 
            return 1;
        }
    }
    else
    {
        try{start.load(sfile);}catch(...){cout<<"Starting values not read."<<endl; return 1;}
    }
    results=irtmle(order, mparams,
                     algorithm,  patterns, 
                     patternnumber, patno,
                     data, thetas, thetano, 
                     scale, scaleno, obsscale, 
                     selectbeta, selectbetano, selbetano,
                     selectbetac, selectbetacno, selbetacno,
                     selectthetai, selectthetaino, selthetaino,
                     selectthetad, selectthetadno, selthetadno,
                     selectthetac, selectthetacno, selthetacno,
                     w, wno,  obssel, obsselno, 
                     obsweight, datasel,
                     betasel, betaselno, start);
    savmaxf2v(order,results,outfile,fflag,pflag);
    return 0;
}
