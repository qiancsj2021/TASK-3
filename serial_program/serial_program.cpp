#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <vector>
#define _USE_MATH_DEFINES
#include <math.h>
#include <string>
#include <time.h>

#define PRINT 1000
const double A1 = 0.0;
const double A2 = 2.0;
const double B1 = 0.0;
const double B2 = 1.0;
const double EPS = 5e-6;

using namespace std;

//--------------Part 1: Some necessary calculations----------------//
double u(int i, int j, double h_x, double h_y) {
	return (1.0 + cos(M_PI * (A1 + i * h_x) * (B1 + j * h_y)));
}

//
double k(double x, double y) {
	return (4 + x + y);
}

// Right border: \psi_R = \psi_{M,j}
double psiR(double y) {
	return (6.0 + y) * (-M_PI * y * sin(2 * M_PI * y)) + 1.0 + cos(2 * M_PI * y);
}

// Upper border: \psi_T = \psi_{i,N}
double psiT(double x) {
	return (5.0 + x) * (-M_PI * x * sin(M_PI * x)) + 1.0 + cos(M_PI * x);
}

// Upper right corner
double psiMN(double x, double y, double h_x, double h_y) {
	return (h_x * psiT(x) + h_y * psiR(y)) / (h_x + h_y);
}

//
double F(double x, double y) {
	return M_PI * M_PI * y * y * (4 + x + y) * cos(M_PI * x * y) + M_PI * M_PI * x * x * (4 + x + y) * cos(M_PI * x * y) + M_PI * y * sin(M_PI * x * y) + M_PI * x * sin(M_PI * x * y);
}

// Auxiliary Calculation Coefficients
double a_ij(int i, int j, double h_x, double h_y) {
	return (4 + A1 + i * h_x - 0.5 * h_x + B1 + j * h_y);
}

// b_{i,j}
double b_ij(int i, int j, double h_x, double h_y) {
	return (4 + A1 + i * h_x + B1 + j * h_y - 0.5 * h_y);
}

// c_{i,j}
double c_ij(int i, int j, double h_x, double h_y) {
	double d_x, d_y;
	d_x = A1 + i * h_x; d_y = B1 + j * h_y;
	double hx_2 = h_x * h_x, hy_2 = h_y * h_y;

	return ((k(d_x - 0.5 * h_x, d_y) + k(d_x + 0.5 * h_x, d_y)) / hx_2 + (k(d_x, d_y - 0.5 * h_y) + k(d_x, d_y + 0.5 * h_y)) / hy_2);
}

// d_{i,N}
double d_ij(int i, int j, double h_x, double h_y) {
	double d_x, d_y;
	d_x = A1 + i * h_x; d_y = B1 + j * h_y;
	double hx_2 = h_x * h_x, hy_2 = h_y * h_y;

	return (2.0 * k(d_x, d_y - 0.5 * h_y) / hy_2 + 2.0 / h_y + (k(d_x - 0.5 * h_x, d_y) + k(d_x + 0.5 * h_x, d_y)) / hx_2);
}

// e_{M,j}
double e_ij(int i, int j, double h_x, double h_y) {
	double d_x, d_y;
	d_x = A1 + i * h_x; d_y = B1 + j * h_y;
	double hx_2 = h_x * h_x, hy_2 = h_y * h_y;

	return (2.0 * k(d_x - 0.5 * h_x, d_y) / hx_2 + 2.0 / h_x + (k(d_x, d_y - 0.5 * h_y) + k(d_x, d_y + 0.5 * h_y)) / hy_2);
}

// f_{M,N}
double f_ij(int i, int j, double h_x, double h_y) {
	double d_x, d_y;
	d_x = A1 + i * h_x; d_y = B1 + j * h_y;
	double hx_2 = h_x * h_x, hy_2 = h_y * h_y;

	return (2.0 * k(d_x - 0.5 * h_x, d_y) / hx_2 + 2.0 * k(d_x, d_y - 0.5 * h_y) / hy_2 + 2.0 / h_x + 2.0 / h_y);
}

//------------------Part 2: Necessary function implementations for iterative algorithms------------------------//

// Initialize the right side of the equation: matrix B
void InitializeMatrixB(double* B, int M, int N, double h_x, double h_y) {
	int i, j;
	double d_x, d_y;
	double hx_2 = h_x * h_x, hy_2 = h_y * h_y;

	for (i = 0; i < M; i++) {
		for (j = 0; j < N; j++) {
			d_x = A1 + ((double)i + 1.0) * h_x;	
			d_y = B1 + ((double)j + 1.0) * h_y;

			// lower left corner
			if (i == 0 && j == 0) {
				B[i * N + j] = F(d_x, d_y) + 2.0 * k(d_x - 0.5 * h_x, d_y) / hx_2 + 2.0 * k(d_x, d_y - 0.5 * h_y) / hy_2;
			}
			// left border
			else if (i == 0 && j >= 1 && j <= N - 2) {
				B[i * N + j] = F(d_x, d_y) + 2.0 * k(d_x - 0.5 * h_x, d_y) / hx_2;
			}
			// upper left corner
			else if (i == 0 && j == N - 1) {
				B[i * N + j] = F(d_x, d_y) + 2 * psiT(d_x) / h_y + 2.0 * k(d_x - 0.5 * h_x, d_y) / hx_2;
			}
			// lower border
			else if (i >= 1 && i <= M - 2 && j == 0) {
				B[i * N + j] = F(d_x, d_y) + 2.0 * k(d_x, d_y - 0.5 * h_y) / hy_2;
			}
			// upper border
			else if (i >= 1 && i <= M - 2 && j == N - 1) {
				B[i * N + j] = F(d_x, d_y) + 2.0 * psiT(d_x) / h_y;
			}
			// lower right corner
			else if (i == M - 1 && j == 0) {
				B[i * N + j] = F(d_x, d_y) + 2.0 * psiR(d_y) / h_x + 2.0 * k(d_x, d_y - 0.5 * h_y) / hy_2;
			}
			// right border
			else if (i == M - 1 && j >= 1 && j <= N - 2) {
				B[i * N + j] = F(d_x, d_y) + 2.0 * psiR(d_y) / h_x;
			}
			// upper right corner
			else if (i == M - 1 && j == N - 1) {
				B[i * N + j] = F(d_x, d_y) + (2.0 / h_x + 2.0 / h_y) * psiMN(d_x, d_y, h_x, h_y);
			}
			// inner points
			else {
				B[i * N + j] = F(d_x, d_y);
			}
		}
	}
}

// Calculate dot product
double DotProduct(double* u, double* v, int M, int N, double h_x, double h_y) {
	double res = 0.0;
	for (int i = 0; i <= M - 1; i++) {
		for (int j = 0; j <= N - 1; j++) {
			// Discuss and determine the value of rho_i and rho_j according to the position of the point
			if (i == M - 1 && j == N - 1) {
				res += 0.25 * u[i * N + j] * v[i * N + j];
			}
			else if (i >= 0 && i <= M - 2 && j >= 0 && j <= N - 2) {
				res += u[i * N + j] * v[i * N + j];
			}
			else {
				res += 0.5 * u[i * N + j] * v[i * N + j];
			}
		}
	}
	return res * h_x * h_y;
}

// Calculation norm
double NormVector(double* u, int M, int N, double h_x, double h_y) {
	return sqrt(DotProduct(u, u, M, N, h_x, h_y));
}

// Calculate vector Aw: MN*1 §Ú Ar: MN*1
void CalculateAw(double* Aw, double* w, int M, int N, double h_x, double h_y) {
	int i, j;
	double hx_2 = h_x * h_x, hy_2 = h_y * h_y;

	for (i = 0; i <= M - 1; i++) {
		for (j = 0; j <= N - 1; j++) {
			// lower left corner
			if (i == 0 && j == 0) {
				Aw[i * N + j] = c_ij(1, 1, h_x, h_y) * w[0] - (b_ij(1, 2, h_x, h_y) / hy_2) * w[1] - (a_ij(2, 1, h_x, h_y) / hx_2) * w[N];
			}
			// left border
			else if (i == 0 && j >= 1 && j <= N - 2) {
				Aw[i * N + j] = -(b_ij(1, j + 1, h_x, h_y) / hy_2) * w[j - 1] + c_ij(1, j + 1, h_x, h_y) * w[j] - (b_ij(1, j + 2, h_x, h_y) / hy_2) * w[j + 1] - (a_ij(2, j + 1, h_x, h_y) / hx_2) * w[N + j];
			}
			// upper left corner
			else if (i == 0 && j == N - 1) {
				Aw[i * N + j] = -(2.0 * b_ij(1, N, h_x, h_y) / hy_2) * w[N - 2] + d_ij(1, N, h_x, h_y) * w[N - 1] - (a_ij(2, N, h_x, h_y) / hx_2) * w[N + N - 1];
			}
			// lower border
			else if (i >= 1 && i <= M - 2 && j == 0) {
				Aw[i * N + j] = -(a_ij(i + 1, 1, h_x, h_y) / hx_2) * w[(i - 1) * N] + c_ij(i + 1, 1, h_x, h_y) * w[i * N] - (b_ij(i + 1, 2, h_x, h_y) / hy_2) * w[i * N + 1] - (a_ij(i + 2, 1, h_x, h_y) / hx_2) * w[(i + 1) * N];
			}
			// upper border
			else if (i >= 1 && i <= M - 2 && j == N - 1) {
				Aw[i * N + j] = -(a_ij(i + 1, N, h_x, h_y) / hx_2) * w[(i - 1) * N + N - 1] - (2.0 * b_ij(i + 1, N, h_x, h_y) / hy_2) * w[i * N + N - 2] + d_ij(i + 1, N, h_x, h_y) * w[i * N + N - 1] - (a_ij(i + 2, N, h_x, h_y) / hx_2) * w[(i + 1) * N + N - 1];
			}
			// lower right corner
			else if (i == M - 1 && j == 0) {
				Aw[i * N + j] = -(2.0 * a_ij(M, 1, h_x, h_y) / hx_2) * w[(M - 2) * N] + e_ij(M, 1, h_x, h_y) * w[(M - 1) * N] - (b_ij(M, 2, h_x, h_y) / hy_2) * w[(M - 1) * N + 1];
			}
			// right border
			else if (i == M - 1 && j >= 1 && j <= N - 2) {
				Aw[i * N + j] = -(2.0 * a_ij(M, j + 1, h_x, h_y) / hx_2) * w[(M - 2) * N + j] - (b_ij(M, j + 1, h_x, h_y) / hy_2) * w[(M - 1) * N + j - 1] + e_ij(M, j + 1, h_x, h_y) * w[(M - 1) * N + j] - (b_ij(M, j + 2, h_x, h_y) / hy_2) * w[(M - 1) * N + j + 1];
			}
			// upper right corner
			else if (i == M - 1 && j == N - 1) {
				Aw[i * N + j] = -(2.0 * a_ij(M, N, h_x, h_y) / hx_2) * w[(M - 2) * N + N - 1] - (2.0 * b_ij(M, N, h_x, h_y) / hy_2) * w[M * N - 2] + f_ij(M, N, h_x, h_y) * w[M * N - 1];
			}
			// inner points
			else {
				Aw[i * N + j] = -(a_ij(i + 1, j + 1, h_x, h_y) / hx_2) * w[(i - 1) * N + j] - (b_ij(i + 1, j + 1, h_x, h_y) / hy_2) * w[i * N + j - 1] + c_ij(i + 1, j + 1, h_x, h_y) * w[i * N + j] - (b_ij(i + 1, j + 2, h_x, h_y) / hy_2) * w[i * N + j + 1] - (a_ij(i + 2, j + 1, h_x, h_y) / hx_2) * w[(i + 1) * N + j];
			}
		}
	}
}

// Calculate the residual vector r^(k) = Aw^(k) - B and Calculate the difference between two adjacent iterations w^(k+1) - w^(k)
void Diff2Vector(double* res, double* u, double* v, int M, int N) {
	int i, j;
	for (i = 0; i <= M - 1; i++) {
		for (j = 0; j <= N - 1; j++) {
			res[i * N + j] = u[i * N + j] - v[i * N + j];
		}
	}
}

// Calculate the value of the (k+1)th iteration w^{k+1}_{ij}
void Update_w(double* w, double* pre_w, double tau, double* r, int M, int N) {
	int i, j;
	for (i = 0; i <= M - 1; i++) {
		for (j = 0; j <= N - 1; j++) {
			w[i * N + j] = pre_w[i * N + j] - tau * r[i * N + j];
		}
	}
}

// w^{k}_{ij} = w^{k+1}_{ij}
void Update_pre_w(double* pre_w, double* w, int M, int N) {
	int i, j;
	for (i = 0; i <= M - 1; i++) {
		for (j = 0; j <= N - 1; j++) {
			pre_w[i * N + j] = w[i * N + j];
		}
	}
}

void PrintVecToMat(double* A, int M, int N) {
	int i, j;
	for (i = 0; i < M; i++) {
		for (j = 0; j < N; j++) {
			cout << A[i * N + j] << " ";
		}
		cout << endl;
	}
}

int main(int argc, char* argv[]) {

	// Steps of the grid
	int M = 1000; //atoi(argv[1]);
	int N = 1000; //atoi(argv[2]);
	int size = M * N;
	double h_x = (double)((A2 - A1) / M);
	double h_y = (double)((B2 - B1) / N);

	// Allocate memory for vectors
	double* w = new double[size];
	double* diff_w = new double[size];
	double* pre_w = new double[size];
	double* r = new double[size];
	double* B = new double[size];
	double* Aw = new double[size];
	double* Ar = new double[size];

	double tau;
	clock_t start, end;
	start = clock();

	cout << "Start solving... " << endl << endl;
	// Given an initial iteration value
	for (int i = 0; i <= M * N - 1; i++) {
		w[i] = 0.82;
	}

	bool is_not_solved = true;
	int num_iter = 0;
	double error_norm;

	InitializeMatrixB(B, M, N, h_x, h_y);

	//------------Part 3: Implementation of Iterative Algorithms-----------------//
	while (is_not_solved) {
		Update_pre_w(pre_w, w, M, N);

		// Aw^{k}
		CalculateAw(Aw, pre_w, M, N, h_x, h_y);
		// r^{k} = Aw^{k} - B
		Diff2Vector(r, Aw, B, M, N);
		// Ar^{k}
		CalculateAw(Ar, r, M, N, h_x, h_y);
		// [Ar^{k}, r^{k}] / ||Ar^{k}||^{2}
		tau = DotProduct(Ar, r, M, N, h_x, h_y) / DotProduct(Ar, Ar, M, N, h_x, h_y);

		// w^{k+1}_{ij}
		Update_w(w, pre_w, tau, r, M, N);
		num_iter++;

		// w^(k+1) - w^(k)
		Diff2Vector(diff_w, w, pre_w, M, N);
		error_norm = NormVector(diff_w, M, N, h_x, h_y);
		if (error_norm < EPS) {
			is_not_solved = false;
		}
		if(num_iter % PRINT == 0)
			cout << "n_iters = " << num_iter << " norm = " << error_norm << endl;
	}
	

	// Record the results(the first column is the real value, the second column is the obtained numerical solution
	string file_name = "./serial_results/M" + to_string(M) + "_N" + to_string(N) + ".txt";
	ofstream res_file(file_name, ios::out);
	res_file << M << "," << N << "\n";
	for (int i = 0; i < M; i++){
		for (int j = 0; j < N; j++){
			res_file << u(i + 1, j + 1, h_x, h_y) << "," << w[i * N + j] << "\n";
		}
	}
	res_file.close();


	end = clock();
	cout << "\n\nWhen M = " << M << ", N = " << N << ":" << endl;
	cout << "after " << ((double)end - (double)start) / CLOCKS_PER_SEC << " seconds, the problem is solved!" << endl;
	cout << "Number of iterations = " << num_iter << ", error_norm = " << error_norm << ", eps = " << EPS << "." << endl;

	delete[] w;
	delete[] diff_w;
	delete[] pre_w;
	delete[] r;
	delete[] B;
	delete[] Aw;
	delete[] Ar;

	return 0;
}
















