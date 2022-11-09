//Rescale for adaptive quadrature transformation for integration of function on vector space.
//Transformation of points is loc+lt*point for the vector loc
//and the lower triangular
//matrix lt.
#include<armadillo>
using namespace std;
using namespace arma;
struct maxf2v
{
    vec locmax;
    double max;
    vec grad;
    mat hess;
};
struct vecmat
{
    vec v;
    mat m;
};
maxf2v quadmax(const vector<vec> & , const vec & );
vecmat rescale(const vector<vec> & points, const vec & values)
{ 
    vecmat result;
    int d,i,q;
    q=points.size();
    d=points[0].size();
    maxf2v quadfit;
    quadfit.locmax.resize(d);
    quadfit.hess.resize(d,d);
    quadfit.grad.resize(d);
    result.v.resize(d);
    result.m.resize(d,d); 
    result.v.zeros();
    result.m=eye(d,d);      
    quadfit=quadmax(points,values); 
    if(isnan(quadfit.max)) return result;
    result.v =quadfit.locmax;
    result.m=chol(inv_sympd(-quadfit.hess),"lower");
    return result;
}
