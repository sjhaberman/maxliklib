//Newton-Raphson iteration.
//Point is x0.  At x0, first derivative is der10, and second derivative is der20.
double newtons(double x,double der1,double der2)
{
    if(der2<0.0)
    {
        return (x-der1/der2);
    }
    else
    {
        return (x+der1);
    }
    
}


