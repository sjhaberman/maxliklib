//Transformation of integration points by transformation loc+scale*point
//and dividing integration weights by scale.
#include<armadillo>
using namespace arma;
using namespace std;
struct pw
{
    vec points;
    vec weights;
};
pw adapt(const double & loc, const double & scale, const pw & pws)
{
    int n;
    pw results;
    n=pws.points.n_elem;
    results.points.set_size(n);
    results.weights.set_size(n);
    results.points=loc+scale*pws.points;
    results.weights=pws.weights/scale;
    return results;
}

