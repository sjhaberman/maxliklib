//Find inverse of quantal transformation.
#include<cmath>
using namespace std;

extern char choice;


double invnorm(double);

double invquantal(double p)
{
    switch (choice)
    {
        case 'C': return log(-log(1.0-p));
        case 'L': return log(p/(1.0-p));
        case 'M': return log(p);
        default: return invnorm(p);
    }
}
