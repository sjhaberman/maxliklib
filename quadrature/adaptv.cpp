//Adaptive quadrature transformation for integration of function on vector space.
//Transformation of points is loc+lt*point for the vector loc and the lower triangular
//matrix lt.  Weight are multiplied by the determinant of lt.
#include<armadillo>
using namespace arma;
using namespace std;
struct pwv
{
    mat points;
    vec weights;
};

pwv adaptv(vec loc,mat lt,pwv pws)

{
    double scale;
    int i;
    vec even;
    pwv results=pws;
    if(min(diagvec(lt))<=0.0)return results;
    scale=prod(diagvec(lt));
    for(i=0;i<results.points.n_cols;i++)
    {
        results.points.col(i)=loc+lt*pws.points.col(i);
    }
    results.weights=scale*pws.weights;
    
    
    return results;
}

