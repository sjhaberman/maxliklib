//Log likelihood component and its first and second derivative
//for probit model with response y and parameter beta.


extern char choice;
struct fd2
{
    double value;
    double der1;
    double der2;
};
fd2 cloglog(int,double);
fd2 logit(int,double);
fd2 logmean(int,double);
fd2 probit(int,double);

fd2 quantal(int y,double beta)
{
    switch (choice)
    {
        case 'C': return cloglog(y,beta);
        case 'L': return logit(y,beta);
        case 'M': return logmean(y,beta);
        default: return probit(y,beta);
    }
}
