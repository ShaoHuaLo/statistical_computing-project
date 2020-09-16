/*
 * main.cpp
 */


#define GETREG 1
#define SHUTDOWNTAG 0

#include<mpi.h>
#include<gsl/gsl_eigen.h>
#include<gsl/gsl_matrix.h>
#include<gsl/gsl_vector.h>
#include<gsl/gsl_randist.h>
#include<gsl/gsl_rng.h>
#include<gsl/gsl_blas.h>
#include<gsl/gsl_linalg.h>
#include<gsl/gsl_statistics_double.h>
#include<gsl/gsl_permutation.h>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include "newregmodels.h"

static int myrank;

void primary(int p);
void replica(int mystatus, int nsim, gsl_matrix* data);
double getMonteCarloInt(int i, gsl_matrix* data, gsl_rng* RNG, int nsim);
double stableLogMean(const gsl_vector* logdata);
void getMCMC(int i, gsl_matrix* data, gsl_matrix* negHessian, double* betaHat, double* betaHatMH, gsl_rng* RNG, int nsim);
void getMVNsample(gsl_matrix* cholesky, gsl_vector* output, gsl_rng* RNG, gsl_vector* tempvec);
double getLaplace(int i, gsl_matrix* data, gsl_matrix* negHessian, double* betaHat);
void getMLE(int i, double* betaHat, gsl_matrix* data, gsl_matrix* negHessian);
void getPi(gsl_vector* x, gsl_vector* beta, gsl_vector* Pi);
void getPi2(gsl_vector* x, gsl_vector* beta, gsl_vector* Pi2);
double Lstar(gsl_vector* y, gsl_vector* Pi, gsl_vector* beta);
void getGradient(gsl_vector* y, gsl_vector* x, gsl_vector* Pi, gsl_vector* beta, gsl_vector* gradient, gsl_vector* ones, gsl_vector* workspace);
void getNegHessian(gsl_vector* x, gsl_vector* Pi2, gsl_matrix* negHessian, gsl_vector* ones, gsl_vector* workspace);
double LogDeterminant(gsl_matrix* A);


int main(int argc, char* argv[])
{
	int nsim = atoi(argv[1]);
	int n = 148, p = 61;
	char file[] = "534finalprojectdata.txt";
	FILE* datastream = fopen(file, "r");
	gsl_matrix* data = gsl_matrix_calloc(n, p);

	/* Each function reads in the data set */
	if(0 != gsl_matrix_fscanf(datastream, data))
	{
		fprintf(stderr, "Failed to read matrix from file [%s] \n", file);
		return(0);
	}
	fclose(datastream);
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

	if(myrank == 0)
	{
		primary(p);
	}
	else
	{
		replica(myrank, nsim, data);
	}
	MPI_Finalize();
	return(1);
}


void primary(int p)
{
	int nreplicas;
	int nworking = 1;
	int i, rank;
	int work[1];
	double results[7];
	MPI_Status status;
	LPRegression Regs = new Regression;
	Regs->Next = NULL;

	LPRegression temp;

	printf("Beginning task distribution. \n");

	// get number of available replicas
	MPI_Comm_size(MPI_COMM_WORLD, &nreplicas);
	for(i = 0; i < p-1; i++)
	{
		// start off by giving each replica a task, namely the variable number
		if(nworking < nreplicas)
		{
					MPI_Send(&i,
					1,
					MPI_INT,
					nworking,
					GETREG,
					MPI_COMM_WORLD);
			nworking++;
		}
		// each replica is working, so we wait until one finishes, get its results,
		// and give it a new task.
		else
		{
			MPI_Recv(results,
					 7,
					 MPI_DOUBLE,
					 MPI_ANY_SOURCE,
					 MPI_ANY_TAG,
					 MPI_COMM_WORLD,
					 &status);

			// the first result is the matrix column index.
			// Add one to convert it back to the variable number
			work[0] = (int)results[0] + 1;
			printf("The primary process has received results for variable %d (ML = %.3lf)\n",
					work[0], results[2]);

			// results[1] = Laplace approx
			// results[2] = MC approx
			// results[3],[4] = beta0, beta1.

			// we try adding the new regression to our list, which maintains
			// the top 5 regressions.
			AddRegression(5, Regs, 1, &work[0], results[2], results[1],
					results[3], results[4]);

			MPI_Send(&i,
					1,
					MPI_INT,
					status.MPI_SOURCE,
					GETREG,
					MPI_COMM_WORLD);
		}
	}

/* retrieve final results from the replica processes */
	for(i = 1; i < nworking; i++)
	{
		MPI_Recv(results,
				7,
				MPI_DOUBLE,
				MPI_ANY_SOURCE,
				MPI_ANY_TAG,
				MPI_COMM_WORLD,
				&status);
		// save results to the output vector, see above for specifics
		work[0] = (int)results[0] + 1;
		printf("The primary process has received results for variable %d (ML = %.3lf)\n",
				work[0], results[2]);
		AddRegression(5, Regs, 1, &work[0], results[2], results[1],
				results[3], results[4]);
	}

	/* we now output all out results to the terminal in an orderly fashion*/
	printf("\nThe top five regressions by marginal likelihood are listed below: \n\n");
	printf("Index: index of predictor variable.\n");
	printf("MC Lik: log marginal likelihood, estimated by\n\tMonte Carlo integration.\n");
	printf("Lap Lik: log marginal likelihood, estimated by\n\tthe Laplace transformation.\n");
	printf("Beta0: posterior mean of Beta0, estimated by\n\tMarkov chain Monte Carlo.\n");
	printf("Beta1: posterior mean of Beta1, estimated by\n\tMarkov chain Monte Carlo.\n\n");

	temp = Regs->Next;
	while(temp != NULL)
	{
		printf("Index\tMC Lik\tLap Lik\tBeta0\tBeta1\n");
		printf("%d\t%.3lf\t%.3lf\t%.3lf\t%.3lf\n\n",
				temp->A[0],temp->logmarglikA,temp->logmarglikLap,temp->beta0, temp->beta1);
		temp = temp->Next;
	}

	DeleteAllRegressions(Regs);
	for(rank=1; rank < nreplicas; rank++)
	{
		MPI_Send(0,
				0,
				MPI_INT,
				rank,
				SHUTDOWNTAG,
				MPI_COMM_WORLD);
	}
	return;
}



void replica(int mystatus, int nsim, gsl_matrix* data)
{
	int i, j;
	int notDone = 1;
	MPI_Status status;

	const gsl_rng_type* T;
	T = gsl_rng_default;
	gsl_rng* RNG;
	RNG = gsl_rng_alloc(T);

	// burn in the generator on each replica process
	for(j = 0; j < 2*nsim*mystatus; j++)
	{
		gsl_rng_uniform(RNG);
	}

	while(notDone)
	{
		MPI_Recv(&i,
				 1,
				 MPI_INT,
				 0,
				 MPI_ANY_TAG,
				 MPI_COMM_WORLD,
				 &status);
		switch(status.MPI_TAG)
		{
		case GETREG:
		{
		/* the replica process has been asked to compute output for
		 * variable i.
		 */
			double betaHat[2];
			double betaHatMH[2];
			double laplace, MCint;
			double results[7];

			// this Hessian matrix will be carried throughout computations.
			gsl_matrix* negHessian = gsl_matrix_alloc(2,2);

			// the next four functions compute the quantities of interest.
			getMLE(i, betaHat, data, negHessian);
			laplace = getLaplace(i, data, negHessian, betaHat);// return type double
			getMCMC(i, data, negHessian, betaHat, betaHatMH, RNG, nsim);
			MCint = getMonteCarloInt(i, data, RNG, nsim);

			results[0] = (double)i;
			results[1] = laplace;
			results[2] = MCint;
			results[3] = betaHatMH[0];
			results[4] = betaHatMH[1];
			results[5] = betaHat[0];
			results[6] = betaHat[1];

			/* send our output back to the primary process */
			MPI_Send(&results,
					 7,
					 MPI_DOUBLE,
					 0,
					 0,
					 MPI_COMM_WORLD);
			gsl_matrix_free(negHessian);
			break;
		}
		default:
			notDone = 0;
		} //close switch
	}
	gsl_rng_free(RNG);
	return;
}

/* this function computes the monte carlo approximation to the marginal
 * likelihood.
 */
double getMonteCarloInt(int i, gsl_matrix* data, gsl_rng* RNG, int nsim)
{
	int k;
	double v;
	int n = data->size1;
	int p = data->size2;
	gsl_vector_view x,y;
	y = gsl_matrix_column(data, p - 1);
	x = gsl_matrix_column(data, i);
	gsl_vector* output = gsl_vector_alloc(nsim);

	//used for sampling MVN
	gsl_vector* proposal = gsl_vector_alloc(2);
	gsl_vector* propPi = gsl_vector_alloc(n);
	gsl_matrix* Identity = gsl_matrix_alloc(2,2);
	gsl_matrix_set_identity(Identity);
	gsl_vector* tempvec = gsl_vector_alloc(Identity->size2);

	for(k = 0; k < nsim; k++)
	{
		getMVNsample(Identity, proposal, RNG, tempvec);
		getPi(&x.vector, proposal, propPi);

		// we want to use L, not Lstar, so we have to add some stuff back.
		v = Lstar(&y.vector, propPi, proposal) + log(2*M_PI) +
				.5*(pow(gsl_vector_get(proposal,0),2) + pow(gsl_vector_get(proposal,1),2));
		gsl_vector_set(output, k, v);
	}

	// we take the mean of the observations, on the log scale
	v = stableLogMean(output);

	gsl_vector_free(tempvec);
	gsl_vector_free(proposal);
	gsl_vector_free(propPi);
	gsl_vector_free(output);
	gsl_matrix_free(Identity);

	return(v);
}

// get's the log of the mean of a set of values, when only the log of the
// values is known.
double stableLogMean(const gsl_vector* logdata)
{
	int n = logdata->size, i;
	double max = gsl_vector_max(logdata);
	double mean, logmean = 0;

	for(i = 0; i < n; i++)
	{
		logmean += exp(gsl_vector_get(logdata, i) - max);
	}

	logmean = max - log(n) + log(logmean);

	mean = exp(logmean);
	return(logmean);
}


/*
 * This function retrieves the MCMC estimate of the mean of the posterior
 * distribution.
 */
void getMCMC(int i, gsl_matrix* data, gsl_matrix* negHessian, double* betaHat,
		double* betaHatMH, gsl_rng* RNG, int nsim)
{
	int k;
	int p = negHessian->size1;
	int n = data->size1, signum;

	double a1, curLik, propLik, logU;

	//allocations
	gsl_vector_view y = gsl_matrix_column(data, data->size2 - 1);
	gsl_vector_view x = gsl_matrix_column(data, i);
	gsl_matrix* LU = gsl_matrix_alloc(p,p);
	gsl_permutation* P = gsl_permutation_alloc(p);

	gsl_matrix* outmatrix = gsl_matrix_alloc(nsim,p);
	gsl_vector* current = gsl_vector_alloc(p);
	gsl_vector* curPi = gsl_vector_alloc(n);
	gsl_vector* propPi = gsl_vector_alloc(n);
	gsl_vector_set(current, 0, betaHat[0]);
	gsl_vector_set(current, 1, betaHat[1]);

	// get the cholesky decomposition of the inverse
	// of the hessian
	gsl_matrix_memcpy(LU, negHessian);
	gsl_linalg_LU_decomp(LU, P, &signum);
	gsl_matrix* negHessInverse = gsl_matrix_alloc(p,p);
	gsl_linalg_LU_invert(LU, P, negHessInverse);
	gsl_linalg_cholesky_decomp(negHessInverse);

	gsl_vector* proposal = gsl_vector_alloc(p);
	gsl_vector* tempvec = gsl_vector_alloc(negHessInverse->size2);

	//initialize the chain
	getPi(&x.vector, current, curPi);
	curLik = Lstar(&y.vector, curPi, current);
	for(k = 0; k < nsim; k++)
	{
		//get proposal
		logU = log(gsl_rng_uniform(RNG));
		getMVNsample(negHessInverse, proposal, RNG, tempvec);
		gsl_vector_add(proposal, current);

		getPi(&x.vector, proposal, propPi);
		propLik = Lstar(&y.vector, propPi, proposal);
		a1 = propLik - curLik;

		if(logU <= a1)
		{
			gsl_vector_memcpy(current, proposal);
			curLik = propLik;
		}
		gsl_matrix_set_row(outmatrix, k, current);
	}
	gsl_vector_view b0, b1;
	b0 = gsl_matrix_column(outmatrix, 0);
	b1 = gsl_matrix_column(outmatrix, 1);

	// gsl_stats_mean computes the mean using the recursive method
	// mean_(n) = mean(n-1) + (data[n] - mean(n-1))/(n+1)
	betaHatMH[0] = gsl_stats_mean(b0.vector.data, b0.vector.stride, outmatrix->size1);
	betaHatMH[1] = gsl_stats_mean(b1.vector.data, b1.vector.stride, outmatrix->size1);

	gsl_vector_free(tempvec);
	gsl_vector_free(current);
	gsl_vector_free(proposal);
	gsl_vector_free(curPi);
	gsl_vector_free(propPi);
	gsl_matrix_free(negHessInverse);
	gsl_matrix_free(outmatrix);
	gsl_permutation_free(P);
	gsl_matrix_free(LU);
	return;
}

void getMVNsample(gsl_matrix* cholesky, gsl_vector* output, gsl_rng* RNG, gsl_vector* tempvec)
{
	int p = cholesky->size2, i;
	//tempvec will store the vector of p random N(0,1)'s
	for(i = 0; i < p; i++)
	{
		gsl_vector_set(tempvec, i, gsl_ran_ugaussian(RNG));
	}

	gsl_blas_dgemv(CblasNoTrans, 1.0, cholesky, tempvec, 0.0, output);
	return;
}

double getLaplace(int i, gsl_matrix* data, gsl_matrix* negHessian, double* betaHat)
{
	double out;
	int p = data->size2;
	int n = data->size1;

	gsl_vector* Pi = gsl_vector_alloc(n);
	gsl_vector* beta = gsl_vector_alloc(2);

	gsl_vector_set(beta, 0, betaHat[0]);
	gsl_vector_set(beta, 1, betaHat[1]);

	gsl_vector_view x,y;
	x = gsl_matrix_column(data, i);
	y = gsl_matrix_column(data, p-1);

	getPi(&x.vector , beta, Pi);
	out = log(2*M_PI) + Lstar(&y.vector, Pi, beta) - .5*LogDeterminant(negHessian);

	gsl_vector_free(Pi);
	gsl_vector_free(beta);
	//printf("laplace: %.3lf \n", out);
	return(out);
}

// This function will put the posterior mode of the logistic regression on
// the i`th variable into betaHat,
// and also put the negative hessian evaluated at betaHat into negHessian.
void getMLE(int i, double* betaHat, gsl_matrix* data, gsl_matrix* negHessian)
{
	int p = data->size2;
	int n = data->size1;

	// tolerance for terminating the NR algorithm.
	double epsilon = .0001;

	gsl_vector_view x,y;
	x = gsl_matrix_column(data, i);
	y = gsl_matrix_column(data, p-1);

	gsl_vector* beta = gsl_vector_calloc(2);
	gsl_vector* betadiff = gsl_vector_alloc(2);
	gsl_vector_set_all(betadiff, 1.0);

	gsl_vector* gradient = gsl_vector_calloc(2);
	gsl_vector* Pi = gsl_vector_alloc(n);
	gsl_vector* Pi2 = gsl_vector_alloc(n);

	// these vectors will be used as workspaces for intermediate calculations.
	gsl_vector* workspace = gsl_vector_calloc(n);
	gsl_vector* ones = gsl_vector_alloc(n);
	gsl_vector_set_all(ones, 1.0);

	gsl_matrix* cholesky = gsl_matrix_calloc(2,2);

	// while loop looks at euclidean distance between parameter updates.
	while(gsl_blas_dnrm2(betadiff) > epsilon)
	{
		getPi(&x.vector, beta, Pi);
		getPi2(&x.vector, beta, Pi2);
		getGradient(&y.vector, &x.vector, Pi, beta, gradient, ones, workspace);
		getNegHessian(&x.vector, Pi2, negHessian, ones, workspace);
		gsl_matrix_memcpy(cholesky, negHessian);
		gsl_linalg_cholesky_decomp(cholesky);

		// betadiff holds the difference between the iterations of beta
		gsl_linalg_cholesky_solve(cholesky, gradient, betadiff);

		// already working with the negative hessian, so we add
		// rather than subtract the update.
		gsl_vector_add(beta, betadiff);
	}

	// one last update to the NegHessian so that it is evaluated at the mode.
	getPi(&x.vector, beta, Pi);
	getPi2(&x.vector, beta, Pi2);
	getGradient(&y.vector, &x.vector, Pi, beta, gradient, ones, workspace);
	getNegHessian(&x.vector, Pi2, negHessian, ones, workspace);

	betaHat[0] = gsl_vector_get(beta, 0);
	betaHat[1] = gsl_vector_get(beta, 1);

	gsl_vector_free(beta);
	gsl_vector_free(betadiff);
	gsl_vector_free(gradient);
	gsl_vector_free(Pi);
	gsl_vector_free(Pi2);
	gsl_vector_free(workspace);
	gsl_vector_free(ones);
	gsl_matrix_free(cholesky);

	return;
}

// note the stable arithmetic
void getPi(gsl_vector* x, gsl_vector* beta, gsl_vector* Pi)
{
	int n = x->size, i;
	double p, g;
	for(i = 0; i < n; i++)
	{
		g = gsl_vector_get(beta,1)*gsl_vector_get(x,i) + gsl_vector_get(beta,0);
		p = 1.0/(1.0 + exp(-g));
		gsl_vector_set(Pi, i, p);
	}
	return;
}
// note the stable arithmetic
void getPi2(gsl_vector* x, gsl_vector* beta, gsl_vector* Pi2)
{
	int n = x->size, i;
	double g;
	double p, q;
	for(i = 0; i < n; i++)
	{
		g = gsl_vector_get(beta,1)*gsl_vector_get(x,i) + gsl_vector_get(beta,0);
		p = 1.0/(1.0 + exp(-g));
		q = 1.0/(1.0 + exp(g));
		gsl_vector_set(Pi2, i, p*q);
	}
	return;
}

double Lstar(gsl_vector* y, gsl_vector* Pi, gsl_vector* beta)
{
	double L = 0;
	int n = Pi->size, i;
	for(i = 0; i < n; i++)
	{
		// added this check because occasionally Pi or 1 - Pi was numerically zero
		// for some elements, causing nan values to appear in the calculations.
		if((int)gsl_vector_get(y,i) == 0)
		{
			L += log(1 - gsl_vector_get(Pi,i));
		}
		else
		{
			L += log(gsl_vector_get(Pi,i));
		}
	}

	L += -.5*(pow(gsl_vector_get(beta,0),2) + pow(gsl_vector_get(beta,1),2)) - log(2*M_PI);

	return(L);
}

void getGradient(gsl_vector* y, gsl_vector* x, gsl_vector* Pi, gsl_vector* beta, gsl_vector* gradient, gsl_vector* ones, gsl_vector* workspace)
{
	double v0, v1;
	gsl_vector_memcpy(workspace, y);
	gsl_vector_sub(workspace, Pi); // y_i - pi_i
	gsl_blas_ddot(ones, workspace, &v0); // sum (y_i - pi_i)
	gsl_blas_ddot(workspace, x, &v1); //sum(x_i(y_i - pi_i))

	v0 -= gsl_vector_get(beta,0);
	v1 -= gsl_vector_get(beta,1);

	gsl_vector_set(gradient, 0, v0);
	gsl_vector_set(gradient, 1, v1);
	return;
}

void getNegHessian(gsl_vector* x, gsl_vector* Pi2, gsl_matrix* negHessian, gsl_vector* ones, gsl_vector* workspace)
{
	double H00, H10, H11;
	gsl_blas_ddot(ones, Pi2, &H00);
	gsl_blas_ddot(Pi2, x, &H10);
	gsl_vector_memcpy(workspace, x);
	gsl_vector_mul(workspace, x);
	gsl_blas_ddot(Pi2, workspace, &H11);

	H11 += 1;
	H00 += 1;

	gsl_matrix_set(negHessian, 0, 0, H00);
	gsl_matrix_set(negHessian, 1, 1, H11);
	gsl_matrix_set(negHessian, 1, 0, H10);
	gsl_matrix_set(negHessian, 0, 1, H10);
	return;
}

double LogDeterminant(gsl_matrix* A)
{
	int q;
	double out = 0.0;
	int n = A->size1;
	// create storage and temporary objects
	gsl_vector*	eig = gsl_vector_alloc(n);
	gsl_eigen_symm_workspace* workspace = gsl_eigen_symm_alloc(n);
	gsl_matrix* tempA = gsl_matrix_alloc(n,n);


	if(0 != gsl_matrix_memcpy(tempA, A))
	{
		printf("failed to copy matrix /n");
	}

	// compute the eigenvalues. Need to use the matrix tempA since this function
	// mangles the matrix A.
	if(0 != gsl_eigen_symm(tempA, eig, workspace))
	{
		printf("failed to compute the eigenvalues /n");
	}


	for(q = 0; q < n; q++)
	{
		out += log(gsl_vector_get(eig, q));
	}

	// clean the table
	gsl_eigen_symm_free(workspace);
	gsl_vector_free(eig);
	gsl_matrix_free(tempA);

	return(out);
}

