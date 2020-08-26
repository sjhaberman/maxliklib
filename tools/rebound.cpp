//Reset upper and lower bounds.
//Old point is y. Derivative at y is der, Lower bound is lower,
//and upper bound is upper.
void rebound(const double &y,const double &der,double &lower,double &upper)
{
    if(der>0.0) lower=y;
    if(der<0.0) upper=y;
    return;
}


