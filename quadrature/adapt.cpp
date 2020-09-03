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
pw adapt(double & loc,double & scale,pw & pws)
{
    pw results=pws;
    results.points=loc+scale*pws.points;
    results.weights=pws.weights/scale;
    return results;
}

