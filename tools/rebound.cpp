//Reset upper and lower bounds.
//Old point is y. Derivative at y is der, Lower bound is b.lower,
//and upper bound is b.upper.
struct bounds
{
     double lower;
     double upper;
};
bounds  rebound(const double &y, const double & der, const bounds & b)
{
    bounds bb;
    bb = b;
    if(der>0.0) bb.lower=y;
    if(der<0.0) bb.upper=y;
    return bb;
}
