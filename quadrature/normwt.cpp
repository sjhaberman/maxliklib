//Divide weights in quadrature points and weights by normal density.
//Order is n.
#include<armadillo>
using namespace arma;
struct pw
{
    vec points;
    vec weights;
};
pw normwt(const pw & pwi)
{
    pw result;
    result.points.copy_size(pwi.points);
    result.weights.copy_size(pwi.points);    
    result.points=pwi.points;
    result.weights=pwi.weights/normpdf(pwi.points);
    return result;
}
            
            
            
