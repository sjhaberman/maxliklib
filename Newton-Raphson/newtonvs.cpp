//Newton-Raphson iteration for function maximization.
//Point is x, gradient at x is gradx, and
//negative-definite Hessian matrix at x is hessx.
//If -hessx is not positive-definite or
//the inner product of the Newton-Raphson step
//is less than mult times the squared Euclidean norm
//of the gradient, then gradient ascent is used.
//

#include<armadillo>
using namespace std;
using namespace arma;

vec newtonv(double mult,vec x,vec gradx,mat hessx)
{vec y;
    if (-hessx.is_sympd())
    {
        y=x+solve(-hessx,gradx);
        if(dot(y-x,varx.grad)<mult*dot(varx.grad,varx.grad))y=x-varx.grad;

    }
    else
    {
        y=x-gradx;
    }
    return y;
}
