//Generate products of sets of quadrature points and quadrature weights.
//indices gives the products to use.  pw gives the univariate points and weights.
#include<armadillo>
using namespace arma;
using namespace std;
struct pw
{
    vec points;
    vec weights;
};
struct pwv
{
    mat points;
    vec weights;
};
pwv genprods(const imat & indices, const vector<pw> & pws)
{
    double w;
    int i,j;
    pwv results;
    results.weights.set_size(indices.n_cols);
    results.points.set_size(indices.n_rows,indices.n_cols);
    for(j=0;j<indices.n_cols;j++)
    {
        w=1.0;
        for(i=0;i<indices.n_rows;i++)
        {
            w=w*pws[i].weights(indices(i,j));
            results.points(i,j)=pws[i].points(indices(i,j));
        }
        results.weights(j)=w;
    }
    return results;
}

