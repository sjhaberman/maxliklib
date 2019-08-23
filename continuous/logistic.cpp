//Log likelihood component and its gradient and hessian matrix
//for normal model with response y and parameter vector beta
//with elements beta(1) and beta(2)>0.

#include<armadillo>
using namespace arma;
using namespace std;
struct fd2bv
{
    double value;
    vec grad;
    mat hess;
    bool fin;
};
struct fd2
{
    double value;
    double der1;
    double der2;
};
fd2bv locationscale(double,vec,function <fd2(double)>);
fd2 logistic012(double);
fd2bv logistic(double y,vec beta)
{
    
    fd2bv results;
    results=locationscale(y,beta,logistic012);
    return results;
}
