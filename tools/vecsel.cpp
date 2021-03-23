//Select vector elements and form new vector.
//xselect gives elements to select, and y is the original vector.
#include<armadillo>
using namespace arma;
struct xsel
{
    bool all;
    uvec list;
};
vec vecsel(const xsel & xselect, const vec & y)
{
    int d,i,j;
    if(xselect.all) return y;
    return y.elem(xselect.list);
}
