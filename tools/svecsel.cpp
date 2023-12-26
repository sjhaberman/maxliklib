//Find size of xsel vector.
//xselect gives elements to select, and y is the original vector.
#include<armadillo>
using namespace arma;
struct xsel
{
    bool all;
    uvec list;
};
int svecsel(const xsel & xselect, const vec & y)
{
    int d;
    if(xselect.all)
    {
         d=y.n_elem; 
    }
    else
    {
         d=xselect.list.n_elem;
    }
    return d;
}
