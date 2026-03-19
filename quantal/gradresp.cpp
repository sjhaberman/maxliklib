//Log likelihood component, gradient, and Hessian
//for modified graded response model with response y with integer values
//0 to n and parameter vector beta of dimension n.
//Choice of tranformation is determined by transform, with 'G' for
//complementary log-log, 'H' for log-log,
//'L' for logit, and 'N' for probit.
//If order is 0, only the function is
//found, if order is 1, then the function and gradient are found.
//If order is 2,
//then the function, gradient, and Hessian are returned.
#include<armadillo>
using namespace arma;
struct f2v
{
    double value;
    vec grad;
    mat hess;
};
struct resp
{
    ivec iresp;
    vec dresp;
};
f2v berresp(const int & , const char & , const resp & , const vec & );
f2v gradresp(const int & order, const char & transform ,
    const resp & y, const vec & beta)
{
    double e;
    f2v results, resultp, resultq;
    int i, n, nn, nnn;
    resp z, zz;
    z.iresp={0};
    zz.iresp={1};
    n=beta.n_elem;
    nn=n-y.iresp(0);
    if(order>0)results.grad.zeros(n);
    if(order>1)results.hess.zeros(n,n);
    if(nn==0){
        resultp=berresp(order, transform, zz, beta);
        results.value=resultp.value;
        if(order==0)return results;
        results.grad(0)=resultp.grad(0);
        if(order==1)return results;
        results.hess(0,0)=resultp.hess(0,0);
        return results;
    }
    if(n==1){
        resultp=berresp(order, transform, z, beta);
        results.value=resultp.value;
        if(order==0)return results;
        results.grad(0)=resultp.grad(0);
        if(order==1)return results;
        results.hess(0,0)=resultp.hess(0,0);
        return results;
    }
    nnn=std::min(nn+1,n);
    vec gamma(1), ebeta(nnn), tau(nnn), etau(nnn),febeta(nnn);
    ebeta=exp(beta.head(nnn));
    tau(0)=beta(0);
    etau(0)=ebeta(0);
    for(i=1;i<nnn;i++){
        etau(i)=etau(i-1)+ebeta(i);
        tau(i)=log(etau(i));
    }
    if(order>0)febeta=(1.0/etau(nnn-1))*ebeta;
    if(nn==n){
        gamma(0)=tau(n-1);
        resultp=berresp(order, transform, z, gamma);
        results.value=resultp.value;
        if(order==0)return results;
        results.grad=resultp.grad(0)*febeta;
        if(order==1)return results;
        results.hess=(resultp.hess(0,0)-resultp.grad(0))*febeta*febeta.t()
            +resultp.grad(0)*diagmat(febeta);
        return results;
    }
    else{
        vec d(2);
        gamma(0)=tau(nn-1);
        resultq=berresp(order,transform,zz,gamma);
        d(1)=exp(resultq.value);
        gamma(0)=tau(nn);
        resultp=berresp(order,transform,zz,gamma);
        d(0)=exp(resultp.value);
        e=d(0)-d(1);
        results.value=log(e);
        if(order==0)return results;
        vec f(2),gr(2),febeta1(nn);
        f=(1.0/e)*d;
        gr(0)=f(0)*resultp.grad(0);
        gr(1)=-f(1)*resultq.grad(0);
        febeta1=(1.0/etau(nn-1))*ebeta.head(nn);
        results.grad.head(nnn)=gr(0)*febeta;
        results.grad.head(nn)=results.grad.head(nn)+gr(1)*febeta1;
        if(order==1)return results;
        results.hess.submat(0,0,nn,nn)=f(0)*(resultp.hess(0,0)+resultp.grad(0)*resultp.grad(0))*febeta*febeta.t()
           +gr(0)*(diagmat(febeta)-febeta*febeta.t());
        results.hess.submat(0,0,nn-1,nn-1)=results.hess.submat(0,0,nn-1,nn-1)
            -f(1)*(resultq.hess(0,0)+resultq.grad(0)*resultq.grad(0))*febeta1*febeta1.t()
            +gr(1)*(diagmat(febeta1)-febeta1*febeta1.t());;
        results.hess.submat(0,0,nn,nn)=results.hess.submat(0,0,nn,nn)
            -results.grad.head(nnn)*results.grad.head(nnn).t();
        return results;
    }
}
