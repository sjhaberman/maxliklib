//Find size of xsel vector.
//xselect gives elements to select, and y is the original vector.
#include<armadillo>
using namespace arma;
struct xsel{bool all; uvec list;};
uword svecsel(const xsel & xselect, const vec & y){
    if(xselect.all) return y.n_elem;
    return xselect.list.n_elem;
}
