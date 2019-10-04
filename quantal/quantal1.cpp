//Log likelihood component and its first derivative
//for quantal model with response y and parameter beta.


extern char choice;
struct fd1
{
    double value;
    double der1;
    
};
fd1 cloglog1(int,double);
fd1 logit1(int,double);
fd1 logmean1(int,double);
fd1 probit1(int,double);

fd1 quantal1(int y,double beta)
{
    switch (choice)
    {
        case 'C': return cloglog1(y,beta);
        case 'L': return logit1(y,beta);
        case 'M': return logmean1(y,beta);
        default: return probit1(y,beta);
    }
}
