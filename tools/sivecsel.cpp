//Find number of integer vector elements selected.
//xselect gives elements to select, and y is the original vector.
#include<armadillo>
using namespace arma;
struct xsel{bool all; uvec list;};
uword sivecsel(const xsel & xselect, const ivec & y){
    if(xselect.all) return y.n_elem;
    return xselect.list.n_elem;
}
