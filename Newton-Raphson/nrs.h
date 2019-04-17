//Struct for Newton-Rapshon algorithm for function maximization.
struct fd2s
{
    double value;
    double der1;
    double der2;
    double der2s;
};

struct nrsvar
{
    double locmax;
    double max;
    double der1;
    double der2;
    double der2s;
};
nrsvar nrsvarf(double x,function<fd2s(double)>f)
{
    fd2s resultf;
    nrsvar result;
    resultf=f(x);
    result.locmax=x;
    result.max=resultf.value;
    result.der1=resultf.der1;
    result.der2=resultf.der2;
    result.der2s=resultf.der2s;
    return result;
}
