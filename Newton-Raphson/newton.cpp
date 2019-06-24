//Newton-Raphson iteration.
//Point is x.  At x, first derivative is der1, and second derivative is der2.
double newton(double x,double der1,double der2)
{
    return (x-der1/der2);
}


