//Struct for Newton-Rapshon algorithm for function maximization.  Multivariate case.
struct fd2v
{
    double value;
    vec grad;
    mat hess;
};

struct nrvvar
{
    vec locmax;
    double max;
    vec grad;
    mat hess;
};
nrvvar nrvvarf(vec x,function<fd2v(vec)> f)
{
    fd2v resultf;
    nrvvar result;
    resultf=f(x);
    result.locmax=x;
    result.max=resultf.value;
    result.grad=resultf.grad;
    result.hess=resultf.hess;
    return result;
}

