#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <vector>
#define _USE_MATH_DEFINES
#include <math.h>
#include <time.h>
#include <string>
#include <mpi.h>
#include <omp.h>

const double A1 = 0.0;
const double A2 = 2.0;
const double B1 = 0.0;
const double B2 = 1.0;
const double EPS = 5e-6;
#define PRINT 1000

using namespace std;

// ------------------- Part 1: Decomposing Regions Using MPI Topology-------------------//

// A structure containing the information needed for a process block
struct ProcInfo {

	int num_procs;
	int dims_x, dims_y; // Record how many processes are on the x and y axes

	int rank; // the rank of the process block

	int coords[2]; // The coordinates of this process block in the MPI topology

	int size_x, size_y; // The size of the process block

	// The startand end coordinates of the process block
	int i_beg, i_end;
	int j_beg, j_end;

	// Process block size containing neighbor block boundary information. real_size_x = size_x + 2; real_size_y = size_y + 2
	int real_size_x, real_size_y;

	// The rank of the process block's upper, lower, left, and right neighbors
	int nb_left, nb_right, nb_up, nb_down;
};

// Find the closest two factors a*b=num_procs && min|a-b|
int FindNumproc_X(int num_procs) {
	int s = (int)sqrt(num_procs);
	for (int i = s; i > 0; i--) {
		if (num_procs % i == 0) {
			return i;
		}
	}
}

// Decompose the region and obtain the necessary information about the process block itself
void DomainDecomp(int M, int N, MPI_Comm* Grid_Comm, ProcInfo* Proc) {

	// The MPI library is being activated...
	MPI_Comm_size(MPI_COMM_WORLD, &(Proc->num_procs));
	MPI_Comm_rank(MPI_COMM_WORLD, &(Proc->rank));

	// Determine how many processes are in each of the x, y directions
	int dims[2] = { 0,0 };
	dims[0] = FindNumproc_X(Proc->num_procs);
	dims[1] = Proc->num_procs / dims[0];
	Proc->dims_x = dims[0]; Proc->dims_y = dims[1];

	// Creating MPI Topology...
	const int ndims = 2;
	int periods[2] = { 0,0 };
	MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, 0, Grid_Comm);
	MPI_Comm_rank(*Grid_Comm, &(Proc->rank));
	MPI_Cart_coords(*Grid_Comm, Proc->rank, ndims, Proc->coords);

	// Determine the number of nodes in each domain
	Proc->size_x = M / dims[0];
	Proc->size_y = N / dims[1];
	if (Proc->coords[0] < (M % dims[0]))
		Proc->size_x += 1;
	if (Proc->coords[1] < (N % dims[1]))
		Proc->size_y += 1;

	// Calculate the starting point and ending point of each domain
	// i_beg
	if (Proc->coords[0] < M % dims[0]) {
		Proc->i_beg = Proc->coords[0] * (M / dims[0]) + Proc->coords[0];
	}
	else { // Proc->coords[0] >= M % dims[0]
		Proc->i_beg = Proc->coords[0] * (M / dims[0]) + M % dims[0];
	}
	Proc->i_end = Proc->i_beg + Proc->size_x - 1;

	// j_beg
	if (Proc->coords[1] < N % dims[1]) {
		Proc->j_beg = Proc->coords[1] * (N / dims[1]) + Proc->coords[1];
	}
	else {
		Proc->j_beg = Proc->coords[1] * (N / dims[1]) + N % dims[1];
	}
	Proc->j_end = Proc->j_beg + Proc->size_y - 1;

	// real_size is the process block size with neighbor block boundary information
	Proc->real_size_x = Proc->size_x + 2;
	Proc->real_size_y = Proc->size_y + 2;

	// Get the rank of a neighboring process block
	MPI_Cart_shift(*Grid_Comm, 0, 1, &(Proc->nb_left), &(Proc->nb_right));
	MPI_Cart_shift(*Grid_Comm, 1, 1, &(Proc->nb_down), &(Proc->nb_up));

}

// Exchange process block boundary information
void ExchangeBorder(double* w_plus, MPI_Comm* Grid_Comm, ProcInfo* Proc) {

	int size_x = Proc->real_size_x, size_y = Proc->real_size_y;

	// buffer for receiving and sending
	double* send_up = new double[size_x];
	double* send_down = new double[size_x];
	double* send_left = new double[size_y];
	double* send_right = new double[size_y];

	double* recv_up = new double[size_x];
	double* recv_down = new double[size_x];
	double* recv_left = new double[size_y];
	double* recv_right = new double[size_y];

	int i, j;
	int TAG_X = 0, TAG_Y = 1;

	MPI_Status status;
	MPI_Request request;

	//To avoid deadlock, next, the boundary information will be transmitted along the +Y / -Y / +X / -X directions respectively

	// +Y: When the process block is at the lower border, only send its own upper border information to the upper neighbor; when the process block is at the upper border, only accept the boundary information sent by the lower neighbor. The remaining process blocks both receive and send boundary information
	if ((Proc->nb_down < 0) && (Proc->nb_up >= 0)) { // The process block is at the lower border
		for (i = 1; i <= Proc->size_x; i++) {
			send_up[i] = w_plus[(i + 1) * size_y - 2];
		}
		MPI_Isend(send_up, size_x, MPI_DOUBLE, Proc->nb_up, TAG_Y, *Grid_Comm, &request);
	}
	else if ((Proc->nb_down >= 0) && (Proc->nb_up >= 0)) {
		for (i = 1; i <= Proc->size_x; i++) {
			send_up[i] = w_plus[(i + 1) * size_y - 2];
		}

		MPI_Sendrecv(send_up, size_x, MPI_DOUBLE, Proc->nb_up, TAG_Y, recv_down, size_x, MPI_DOUBLE, Proc->nb_down, TAG_Y, *Grid_Comm, &status);
		for (i = 1; i <= Proc->size_x; i++) {
			w_plus[i * size_y] = recv_down[i];
		}
	}
	else if ((Proc->nb_down >= 0) && (Proc->nb_up < 0)) { // The process block is at the upper border
		MPI_Irecv(recv_down, size_x, MPI_DOUBLE, Proc->nb_down, TAG_Y, *Grid_Comm, &request);
		MPI_Wait(&request, &status);
		for (i = 1; i <= Proc->size_x; i++) {
			w_plus[i * size_y] = recv_down[i];
		}
	}

	// -Y: When the process block is at the upper border, only send its own lower border information to the lower neighbor; when the process block is at the lower border, only accept the boundary information sent by the upper neighbor. The remaining process blocks both receive and send boundary information
	if ((Proc->nb_down >= 0) && (Proc->nb_up < 0)) { // The process block is at the upper border
		for (i = 1; i <= Proc->size_x; i++) {
			send_down[i] = w_plus[i * size_y + 1];
		}
		MPI_Isend(send_down, size_x, MPI_DOUBLE, Proc->nb_down, TAG_Y, *Grid_Comm, &request);
	}
	else if ((Proc->nb_down >= 0) && (Proc->nb_up >= 0)) {
		for (i = 1; i <= Proc->size_x; i++) {
			send_down[i] = w_plus[i * size_y + 1];
		}
		MPI_Sendrecv(send_down, size_x, MPI_DOUBLE, Proc->nb_down, TAG_Y, recv_up, size_x, MPI_DOUBLE, Proc->nb_up, TAG_Y, *Grid_Comm, &status);

		for (i = 1; i <= Proc->size_x; i++) {
			w_plus[(i + 1) * size_y - 1] = recv_up[i];
		}
	}
	else if ((Proc->nb_down < 0) && (Proc->nb_up >= 0)) { // The process block is at the lower border
		MPI_Irecv(recv_up, size_x, MPI_DOUBLE, Proc->nb_up, TAG_Y, *Grid_Comm, &request);
		MPI_Wait(&request, &status);

		for (i = 1; i <= Proc->size_x; i++) {
			w_plus[(i + 1) * size_y - 1] = recv_up[i];
		}
	}

	// +X: When the process block is at the left border, only send its own right border information to the right neighbor; when the process block is at the right border, only accept the boundary information sent by the left neighbor. The remaining process blocks both receive and send boundary information
	if ((Proc->nb_left < 0) && (Proc->nb_right >= 0)) { // The process block is at the left border
		for (j = 1; j <= Proc->size_y; j++) {
			send_right[j] = w_plus[(size_x - 2) * size_y + j];
		}
		MPI_Isend(send_right, size_y, MPI_DOUBLE, Proc->nb_right, TAG_X, *Grid_Comm, &request);
	}
	else if ((Proc->nb_left >= 0) && (Proc->nb_right >= 0)) {
		for (j = 1; j <= Proc->size_y; j++) {
			send_right[j] = w_plus[(size_x - 2) * size_y + j];
		}
		MPI_Sendrecv(send_right, size_y, MPI_DOUBLE, Proc->nb_right, TAG_X, recv_left, size_y, MPI_DOUBLE, Proc->nb_left, TAG_X, *Grid_Comm, &status);
		for (j = 1; j <= Proc->size_y; j++) {
			w_plus[j] = recv_left[j];
		}
	}
	else if ((Proc->nb_left >= 0) && (Proc->nb_right < 0)) { // The process block is at the right border
		MPI_Irecv(recv_left, size_y, MPI_DOUBLE, Proc->nb_left, TAG_X, *Grid_Comm, &request);
		MPI_Wait(&request, &status);
		for (j = 1; j <= Proc->size_y; j++) {
			w_plus[j] = recv_left[j];
		}
	}

	// -X: When the process block is at the right border, only send its own left border information to the left neighbor; when the process block is at the left border, only accept the boundary information sent by the right neighbor. The remaining process blocks both receive and send boundary information
	if ((Proc->nb_left >= 0) && (Proc->nb_right < 0)) { // The process block is at the right border
		for (j = 1; j <= Proc->size_y; j++) {
			send_left[j] = w_plus[size_y + j];
		}
		MPI_Isend(send_left, size_y, MPI_DOUBLE, Proc->nb_left, TAG_X, *Grid_Comm, &request);
	}
	else if ((Proc->nb_left >= 0) && (Proc->nb_right >= 0)) {
		for (j = 1; j <= Proc->size_y; j++) {
			send_left[j] = w_plus[size_y + j];
		}
		MPI_Sendrecv(send_left, size_y, MPI_DOUBLE, Proc->nb_left, TAG_X, recv_right, size_y, MPI_DOUBLE, Proc->nb_right, TAG_X, *Grid_Comm, &status);
		for (j = 1; j <= Proc->size_y; j++) {
			w_plus[(size_x - 1) * size_y + j] = recv_right[j];
		}
	}
	else if ((Proc->nb_left < 0) && (Proc->nb_right >= 0)) { // The process block is at the left border
		MPI_Irecv(recv_right, size_y, MPI_DOUBLE, Proc->nb_right, TAG_X, *Grid_Comm, &request);
		MPI_Wait(&request, &status);
		for (j = 1; j <= Proc->size_y; j++) {
			w_plus[(size_x - 1) * size_y + j] = recv_right[j];
		}
	}

	delete[] send_up; delete[] send_down; delete[] send_left; delete[] send_right;
	delete[] recv_up; delete[] recv_down; delete[] recv_left; delete[] recv_right;
}


//------------------Part 2: Some necessary calculations----------------//
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


//--------------------------Part 3: Iterative algorithm functions to be implemented for each process block----------------------//
// Matrix B corresponding to each process block
void InitializeB(double* B, int M, int N, double h_x, double h_y, ProcInfo* Proc) {
	double hx_2 = h_x * h_x, hy_2 = h_y * h_y;

	// Traverse the nodes contained in the process block

#pragma omp parallel for default(shared) schedule(dynamic)
	for (int i = 1; i <= Proc->size_x; i++) {
		for (int j = 1; j <= Proc->size_y; j++) {

			// Mapping coordinates inside a process block to global coordinates
			int idx_x = i + Proc->i_beg;
			int idx_y = j + Proc->j_beg;

			double d_x = A1 + (double)idx_x * h_x;
			double d_y = B1 + (double)idx_y * h_y;

			// lower left corner
			if (idx_x == 1 && idx_y == 1) {
				B[i * Proc->real_size_y + j] = F(d_x, d_y) + 2.0 * k(d_x - 0.5 * h_x, d_y) / hx_2 + 2.0 * k(d_x, d_y - 0.5 * h_y) / hy_2;
			}
			// left border
			else if (idx_x == 1 && idx_y >= 2 && idx_y <= N - 1) {
				B[i * Proc->real_size_y + j] = F(d_x, d_y) + 2.0 * k(d_x - 0.5 * h_x, d_y) / hx_2;
			}
			// upper left corner
			else if (idx_x == 1 && idx_y == N) {
				B[i * Proc->real_size_y + j] = F(d_x, d_y) + 2.0 * psiT(d_x) / h_y + 2.0 * k(d_x - 0.5 * h_x, d_y) / hx_2;
			}
			// lower border
			else if (idx_x >= 2 && idx_x <= M - 1 && idx_y == 1) {
				B[i * Proc->real_size_y + j] = F(d_x, d_y) + 2.0 * k(d_x, d_y - 0.5 * h_y) / hy_2;
			}
			// upper border
			else if (idx_x >= 2 && idx_x <= M - 1 && idx_y == N) {
				B[i * Proc->real_size_y + j] = F(d_x, d_y) + 2.0 * psiT(d_x) / h_y;
			}
			// lower right corner
			else if (idx_x == M && idx_y == 1) {
				B[i * Proc->real_size_y + j] = F(d_x, d_y) + 2.0 * psiR(d_y) / h_x + 2.0 * k(d_x, d_y - 0.5 * h_y) / hy_2;
			}
			// right border
			else if (idx_x == M && idx_y >= 2 && idx_y <= N - 1) {
				B[i * Proc->real_size_y + j] = F(d_x, d_y) + 2.0 * psiR(d_y) / h_x;
			}
			// upper right corner
			else if (idx_x == M && idx_y == N) {
				B[i * Proc->real_size_y + j] = F(d_x, d_y) + (2.0 / h_x + 2.0 / h_y) * psiMN(d_x, d_y, h_x, h_y);
			}
			// inner points
			else {
				B[i * Proc->real_size_y + j] = F(d_x, d_y);
			}
		}
	}
}

// 
void CalculateAw(double* Aw, double* w, int M, int N, double h_x, double h_y, ProcInfo* Proc) {

	double hx_2 = h_x * h_x, hy_2 = h_y * h_y;

	// Traverse the nodes contained in the process block
#pragma omp parallel for default(shared) schedule(dynamic)
	for (int i = 1; i <= Proc->size_x; i++) {
		for (int j = 1; j <= Proc->size_y; j++) {

			// Mapping coordinates inside a process block to global coordinates
			int idx_x = i + Proc->i_beg;
			int idx_y = j + Proc->j_beg;

			double d_x = A1 + (double)idx_x * h_x;
			double d_y = B1 + (double)idx_y * h_y;

			// lower left corner
			if (idx_x == 1 && idx_y == 1) {
				Aw[i * Proc->real_size_y + j] = c_ij(1, 1, h_x, h_y) * w[1 * Proc->real_size_y + 1] - (b_ij(1, 2, h_x, h_y) / hy_2) * w[1 * Proc->real_size_y + 2] - (a_ij(2, 1, h_x, h_y) / hx_2) * w[2 * Proc->real_size_y + 1];
			}
			// left border
			else if (idx_x == 1 && idx_y >= 2 && idx_y <= N - 1) {
				Aw[i * Proc->real_size_y + j] = -(b_ij(1, idx_y, h_x, h_y) / hy_2) * w[1 * Proc->real_size_y + j - 1] + c_ij(1, idx_y, h_x, h_y) * w[1 * Proc->real_size_y + j] - (b_ij(1, idx_y + 1, h_x, h_y) / hy_2) * w[1 * Proc->real_size_y + j + 1] - (a_ij(2, idx_y, h_x, h_y) / hx_2) * w[2 * Proc->real_size_y + j];
			}
			// upper left corner
			else if (idx_x == 1 && idx_y == N) {
				Aw[i * Proc->real_size_y + j] = -(2.0 * b_ij(1, N, h_x, h_y) / hy_2) * w[1 * Proc->real_size_y + Proc->size_y - 1] + d_ij(1, N, h_x, h_y) * w[1 * Proc->real_size_y + Proc->size_y] - (a_ij(2, N, h_x, h_y) / hx_2) * w[2 * Proc->real_size_y + Proc->size_y];
			}
			// lower border
			else if (idx_x >= 2 && idx_x <= M - 1 && idx_y == 1) { //ÖÐÏÂ
				Aw[i * Proc->real_size_y + j] = -(a_ij(idx_x, 1, h_x, h_y) / hx_2) * w[(i - 1) * Proc->real_size_y + 1] + c_ij(idx_x, 1, h_x, h_y) * w[i * Proc->real_size_y + 1] - (b_ij(idx_x, 2, h_x, h_y) / hy_2) * w[i * Proc->real_size_y + 2] - (a_ij(idx_x + 1, 1, h_x, h_y) / hx_2) * w[(i + 1) * Proc->real_size_y + 1];
			}
			// upper border
			else if (idx_x >= 2 && idx_x <= M - 1 && idx_y == N) {
				Aw[i * Proc->real_size_y + j] = -(a_ij(idx_x, N, h_x, h_y) / hx_2) * w[(i - 1) * Proc->real_size_y + Proc->size_y] - (2.0 * b_ij(idx_x, N, h_x, h_y) / hy_2) * w[i * Proc->real_size_y + Proc->size_y - 1] + d_ij(idx_x, N, h_x, h_y) * w[i * Proc->real_size_y + Proc->size_y] - (a_ij(idx_x + 1, N, h_x, h_y) / hx_2) * w[(i + 1) * Proc->real_size_y + Proc->size_y];
			}
			// lower right corner
			else if (idx_x == M && idx_y == 1) {
				Aw[i * Proc->real_size_y + j] = -(2.0 * a_ij(M, 1, h_x, h_y) / hx_2) * w[(Proc->size_x - 1) * Proc->real_size_y + 1] + e_ij(M, 1, h_x, h_y) * w[(Proc->size_x) * Proc->real_size_y + 1] - (b_ij(M, 2, h_x, h_y) / hy_2) * w[(Proc->size_x) * Proc->real_size_y + 2];
			}
			// right border
			else if (idx_x == M && idx_y >= 2 && idx_y <= N - 1) {
				Aw[i * Proc->real_size_y + j] = -(2.0 * a_ij(M, idx_y, h_x, h_y) / hx_2) * w[(Proc->size_x - 1) * Proc->real_size_y + j] - (b_ij(M, idx_y, h_x, h_y) / hy_2) * w[(Proc->size_x) * Proc->real_size_y + j - 1] + e_ij(M, idx_y, h_x, h_y) * w[(Proc->size_x) * Proc->real_size_y + j] - (b_ij(M, idx_y + 1, h_x, h_y) / hy_2) * w[(Proc->size_x) * Proc->real_size_y + j + 1];
			}
			// upper right corner
			else if (idx_x == M && idx_y == N) {
				Aw[i * Proc->real_size_y + j] = -(2.0 * a_ij(M, N, h_x, h_y) / hx_2) * w[(Proc->size_x - 1) * Proc->real_size_y + Proc->size_y] - (2.0 * b_ij(M, N, h_x, h_y) / hy_2) * w[(Proc->size_x) * Proc->real_size_y + Proc->size_y - 1] + f_ij(M, N, h_x, h_y) * w[(Proc->size_x) * Proc->real_size_y + Proc->size_y];
			}
			// inner points
			else {
				Aw[i * Proc->real_size_y + j] = -(a_ij(idx_x, idx_y, h_x, h_y) / hx_2) * w[(i - 1) * Proc->real_size_y + j] - (b_ij(idx_x, idx_y, h_x, h_y) / hy_2) * w[i * Proc->real_size_y + j - 1] + c_ij(idx_x, idx_y, h_x, h_y) * w[i * Proc->real_size_y + j] - (b_ij(idx_x, idx_y + 1, h_x, h_y) / hy_2) * w[i * Proc->real_size_y + j + 1] - (a_ij(idx_x + 1, idx_y, h_x, h_y) / hx_2) * w[(i + 1) * Proc->real_size_y + j];
			}
		}
	}
}

// 
void Update_pre_w(double* pre_w, double* w, int size_x, int size_y) {
#pragma omp parallel for default(shared) schedule(dynamic)
	for (int i = 0; i < size_x; i++) {
		for (int j = 0; j < size_y; j++) {
			pre_w[i * size_y + j] = w[i * size_y + j];
		}
	}
}

// Calculate the residual vector r^(k) = Aw^(k) - B and Calculate the difference between two adjacent iterations w^(k+1) - w^(k)
void Diff2Vector(double* res, double* u, double* v, int size_x, int size_y) {
#pragma omp parallel for default(shared) schedule(dynamic)
	for (int i = 0; i < size_x; i++) {
		for (int j = 0; j < size_y; j++) {
			res[i * size_y + j] = u[i * size_y + j] - v[i * size_y + j];
		}
	}
}


// Calculate dot product
double DotProduct(double* u, double* v, int M, int N, double h_x, double h_y, ProcInfo* Proc) {

	double res = 0.0;
#pragma omp parallel for default(shared) schedule(dynamic) reduction(+:res)
	// Traverse the nodes contained in the process block
	for (int i = 1; i <= Proc->size_x; i++) {
		for (int j = 1; j <= Proc->size_y; j++) {

			// Mapping coordinates inside a process block to global coordinates
			int idx_x = i + Proc->i_beg;
			int idx_y = j + Proc->j_beg;

			if (idx_x == M && idx_y == N) {
				res += 0.25 * u[i * Proc->real_size_y + j] * v[i * Proc->real_size_y + j];
			}
			else if (idx_x >= 1 && idx_x <= M - 1 && idx_y >= 1 && idx_y <= N - 1) {
				res += u[i * Proc->real_size_y + j] * v[i * Proc->real_size_y + j];
			}
			else {
				res += 0.5 * u[i * Proc->real_size_y + j] * v[i * Proc->real_size_y + j];
			}
		}
	}
	return res * h_x * h_y;
}

// Calculation norm
double NormVector(double* u, double* v, int M, int N, double h_x, double h_y, ProcInfo* Proc) {
	return sqrt(DotProduct(u, u, M, N, h_x, h_y, Proc));
}

// Calculate the value of the (k+1)th iteration w^{k+1}_{ij}
void Update_w(double* w, double* pre_w, double tau, double* r, int M, int N, ProcInfo* Proc) {
#pragma omp parallel for default(shared) schedule(dynamic)
	for (int i = 1; i <= Proc->size_x; i++) {
		for (int j = 1; j <= Proc->size_y; j++) {
			w[i * Proc->real_size_y + j] = pre_w[i * Proc->real_size_y + j] - tau * r[i * Proc->real_size_y + j];
		}
	}
}

// 
void InitMatrix_with_value(double* A, double val, int size_x, int size_y) {
#pragma omp parallel for default(shared) schedule(dynamic)
	for (int i = 0; i < size_x; i++) {
		for (int j = 0; j < size_y; j++) {
			A[i * size_y + j] = val;
		}
	}
}


// Implement an iterative algorithm
double Solve(int M, int N, double h_x, double h_y, MPI_Comm* Grid_Comm, ProcInfo* Proc, int* num_iter) {

	int size_plus = Proc->real_size_x * Proc->real_size_y;

	// Create the necessary arrays (containing information about the boundaries of other process blocks)
	double* w = new double[size_plus];
	double* pre_w = new double[size_plus];
	double* diff_w = new double[size_plus];
	double* r = new double[size_plus];
	double* B = new double[size_plus]; InitMatrix_with_value(B, 0.0, Proc->real_size_x, Proc->real_size_y);
	double* Aw = new double[size_plus]; InitMatrix_with_value(Aw, 0.0, Proc->real_size_x, Proc->real_size_y);
	double* Ar = new double[size_plus]; InitMatrix_with_value(Ar, 0.0, Proc->real_size_x, Proc->real_size_y);

	//
	double tau_global, tau_global_numerator, tau_global_denominator;
	double tau_local_numerator, tau_local_denominator;
	double error_norm;
	double diff_global, diff_local;


	// Given an initial iteration value
	InitMatrix_with_value(w, 0.82, Proc->real_size_x, Proc->real_size_y);

	// Initialize the right side of the equation: matrix B
	InitializeB(B, M, N, h_x, h_y, Proc);

	do {
		// w^(k)
		Update_pre_w(pre_w, w, Proc->real_size_x, Proc->real_size_y);
		// Exchanging boundary information between process blocks
		ExchangeBorder(pre_w, Grid_Comm, Proc);

		// Aw^{k}
		CalculateAw(Aw, pre_w, M, N, h_x, h_y, Proc);

		// r^{k} = Aw^{k} - B
		Diff2Vector(r, Aw, B, Proc->real_size_x, Proc->real_size_y);
		// Exchanging boundary information between process blocks
		ExchangeBorder(r, Grid_Comm, Proc);

		// Ar^{k}
		CalculateAw(Ar, r, M, N, h_x, h_y, Proc);

		// tau
		// step 1. Each process block calculates its own tau numerator and denominator
		tau_local_numerator = DotProduct(Ar, r, M, N, h_x, h_y, Proc);
		tau_local_denominator = DotProduct(Ar, Ar, M, N, h_x, h_y, Proc);

		// step 2. After reduction, compute the public global tau
		MPI_Allreduce(&tau_local_numerator, &tau_global_numerator, 1, MPI_DOUBLE, MPI_SUM, *Grid_Comm);
		MPI_Allreduce(&tau_local_denominator, &tau_global_denominator, 1, MPI_DOUBLE, MPI_SUM, *Grid_Comm);
		tau_global = tau_global_numerator / tau_global_denominator;

		// w^{k+1}_{ij}
		Update_w(w, pre_w, tau_global, r, M, N, Proc);
		*(num_iter) = *(num_iter)+1;

		// ||w^{k+1} - w^{k}|| < EPS
		Diff2Vector(diff_w, w, pre_w, Proc->real_size_x, Proc->real_size_y);

		// Calculate the error within each process block
		diff_local = DotProduct(diff_w, diff_w, M, N, h_x, h_y, Proc);

		// After reduction, compute the public global error
		MPI_Allreduce(&diff_local, &diff_global, 1, MPI_DOUBLE, MPI_SUM, *Grid_Comm);
		error_norm = sqrt(diff_global);

		if (Proc->rank == 0 && *(num_iter) % PRINT == 0) {
			cout << "n_iters = " << *(num_iter) << " norm = " << error_norm << endl;
		}
	} while (error_norm > EPS);


	delete[] Ar; delete[] Aw; delete[] B; delete[] r;
	delete[] diff_w; delete[] pre_w; delete[] w;

	return error_norm;
}


int main(int argc, char* argv[]) {

	int M = atoi(argv[1]); //atoi(argv[1])
	int N = atoi(argv[2]); //atoi(argv[2])

	omp_set_num_threads(4);

	double h_x = (double)((A2 - A1) / M);
	double h_y = (double)((B2 - B1) / N);

	MPI_Init(&argc, &argv);

	double start = MPI_Wtime();

	// A new grid communicator
	ProcInfo Proc;
	MPI_Comm Grid_Comm;
	DomainDecomp(M, N, &Grid_Comm, &Proc);


	if (Proc.rank == 0) {
		cout << "Start solving... " << endl << endl;
	}
	int num_iter = 0;
	double error_norm;

	error_norm = Solve(M, N, h_x, h_y, &Grid_Comm, &Proc, &num_iter);

	MPI_Barrier(MPI_COMM_WORLD);
	double end = MPI_Wtime();

	if (Proc.rank == 0) {
		cout << "\nWhen M = " << M << ", N = " << N << ", num_procs = " << Proc.num_procs << ":" << endl;
		cout << "Number of iterations = " << num_iter << ", error_norm = " << error_norm << ", eps = " << EPS << "." << endl;
		cout << "after " << (end - start) << " seconds, the problem is solved!" << endl;
	}


	MPI_Finalize();
	return 0;
}
