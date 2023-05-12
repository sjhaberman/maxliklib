//Generate products of sets of quadrature points and quadrature weights.
//indices gives the products to use.  pw gives the univariate points and weights.
#include<armadillo>
using namespace std;
using namespace arma;
struct pw
{
    vec points;
    vec weights;
};
struct vecmat
{
    vec v;
    mat m;
};
vecmat genprods(const imat & indices, const vector<pw> & pws)
{
    double w;
    int i,j;
    vecmat results;
    results.v.set_size(indices.n_cols);
    results.m.set_size(indices.n_rows,indices.n_cols);
    for(j=0;j<indices.n_cols;j++)
    {
        w=1.0;
        for(i=0;i<indices.n_rows;i++)
        {
            w=w*pws[i].weights(indices(i,j));
            results.m(i,j)=pws[i].points(indices(i,j));
        }
        results.v(j)=w;
    }
    return results;
}

