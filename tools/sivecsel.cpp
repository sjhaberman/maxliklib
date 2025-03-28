//Find number of integer vector elements selected.
//xselect gives elements to select, and y is the original vector.
#include<armadillo>
using namespace arma;
struct xsel
{
    bool all;
    uvec list;
};
int sivecsel(const xsel & xselect, const ivec & y)
{
    int d;
    if(xselect.all)
    {
        d = y.n_elem;
    }
    else
    {
        d = xselect.list.n_elem;
    }
    return d;
}
