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
fd2 normal012(double);
fd2bv normal(double y,vec beta)
{
    
    fd2bv results;
    results=locationscale(y,beta,normal012);
    return results;
}
