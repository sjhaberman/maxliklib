//Select integer vector elements and form new integer vector.
//xselect gives elements to select, and y is the original vector.
#include<armadillo>
using namespace arma;
struct xsel
{
    bool all;
    uvec list;
};
ivec ivecsel(const xsel & xselect, const ivec & y)
{
    int d,i,j;
    if(xselect.all) return y;
    return y.elem(xselect.list);
}
