//Numerical approximation of gradient.  Original function is f,
//evaluation is at x, and step size is delta.
#include<armadillo>
using namespace std;
using namespace arma;
struct f2v{double value; vec grad; mat hess;};
f2v ng(const double & delta, const vec & x, const function <double(const vec & )>f){
    uword d,i;
    double delta2;
    f2v result;
    result.value=f(x);
    d=x.n_elem;
    result.grad.set_size(d);
    vec u,v;
    u.set_size(d);
    v.set_size(d);
    delta2=delta+delta;
    for(i=0;i<d;i++){
        u=x;
        v=x;
        u(i)=u(i)+delta;
        v(i)=v(i)-delta;
        result.grad(i)=(f(u)-f(v))/delta2;
    }
    return result;
}
