//Reset upper and lower bounds.
//Old point is x. Derivative at x is der, Lower bound is lower, and upper bound is upper.
void rebound(double x,double der,double &lower,double &upper)
{
    if(der>0.0) lower=x;
    if(der<0.0) upper=x;
    return ;
}


