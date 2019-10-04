//Log likelihood component and its first derivative
//for quantal model with response y and parameter beta.


extern char choice;

double cloglog0(int,double);
double logit0(int,double);
double logmean0(int,double);
double probit0(int,double);

double quantal0(int y,double beta)
{
    switch (choice)
    {
        case 'C': return cloglog0(y,beta);
        case 'L': return logit0(y,beta);
        case 'M': return logmean0(y,beta);
        default: return probit0(y,beta);
    }
}
