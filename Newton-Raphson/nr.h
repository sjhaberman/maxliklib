//Struct for Newton-Rapshon algorithm for function maximization.
struct fd2
{
    double value;
    double der1;
    double der2;
};

struct nrvar
{
    double locmax;
    double max;
    double der1;
    double der2;
};
nrvar nrvarf(double x,function <fd2(double)> f)
{
    fd2 resultf;
    nrvar result;
    resultf=f(x);
    result.locmax=x;
    result.max=resultf.value;
    result.der1=resultf.der1;
    result.der2=resultf.der2;
    return result;
}
