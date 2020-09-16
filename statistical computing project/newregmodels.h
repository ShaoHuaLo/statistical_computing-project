/*
 FILE: REGMODELS.H

 Linked list
*/


//this avoids including the function headers twice
#ifndef _REGMODELS
#define _REGMODELS

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct myRegression* LPRegression;
typedef struct myRegression Regression;

struct myRegression
{
	int lenA; //number of regressors
	double logmarglikA; //log marginal likelihood of the regression
	double logmarglikLap; //laplace estimate of the same
	double beta0; //coefficients as estimated by MCMC
	double beta1;
	int* A; //regressors

  LPRegression Next; //link to the next regression
};

void AddRegression(int nMaxRegs, LPRegression regressions, int lenA, int* A,
		double logmarglikA, double laplaceLik, double beta0, double beta1);
void DeleteAllRegressions(LPRegression regressions);
void DeleteLastRegression(LPRegression regressions);
void SaveRegressions(char* filename,LPRegression regressions);

#endif
