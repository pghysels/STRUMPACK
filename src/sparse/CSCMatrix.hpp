/*
 * STRUMPACK -- STRUctured Matrices PACKage, Copyright (c) 2014, The Regents of
 * the University of California, through Lawrence Berkeley National Laboratory
 * (subject to receipt of any required approvals from the U.S. Dept. of Energy).
 * All rights reserved.
 *
 * If you have questions about your rights to use or distribute this software,
 * please contact Berkeley Lab's Technology Transfer Department at TTD@lbl.gov.
 *
 * NOTICE. This software is owned by the U.S. Department of Energy. As such, the
 * U.S. Government has been granted for itself and others acting on its behalf a
 * paid-up, nonexclusive, irrevocable, worldwide license in the Software to
 * reproduce, prepare derivative works, and perform publicly and display publicly.
 * Beginning five (5) years after the date permission to assert copyright is
 * obtained from the U.S. Department of Energy, and subject to any subsequent five
 * (5) year renewals, the U.S. Government is granted for itself and others acting
 * on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
 * Software to reproduce, prepare derivative works, distribute copies to the
 * public, perform publicly and display publicly, and to permit others to do so.
 *
 * Developers: Pieter Ghysels, Francois-Henry Rouet, Xiaoye S. Li.
 *             (Lawrence Berkeley National Lab, Computational Research Division).
 *
 */
#ifndef CSCMATRIX_HPP
#define CSCMATRIX_HPP
#include <sstream>
#include <fstream>
#include <vector>
#include <tuple>

#include "misc/Tools.hpp"
#include "misc/TaskTimer.hpp"
#include "CompressedSparseMatrix.hpp"

namespace strumpack {

  template<typename scalar_t,typename integer_t> class CSCMatrix
    : public CompressedSparseMatrix<scalar_t,integer_t> {
    using DistM_t = DistributedMatrix<scalar_t>;
    using DenseM_t = DistributedMatrix<scalar_t>;
  public:
    CSCMatrix();
    CSCMatrix(integer_t n, integer_t nnz);
    CSCMatrix(integer_t n, integer_t* ptr, integer_t* ind, scalar_t* values, bool symm_sparsity=false);
    CSCMatrix(const CSCMatrix<scalar_t,integer_t>& A);
    CSCMatrix<scalar_t,integer_t>& operator=(const CSCMatrix<scalar_t,integer_t>& A);
    CSCMatrix<scalar_t,integer_t>* clone();
    void spmv(scalar_t* x, scalar_t* y);
    void omp_spmv(scalar_t* x, scalar_t* y);
    void extract_separator(integer_t separator_end, const std::vector<std::size_t>& I,
			   const std::vector<std::size_t>& J, DenseM_t& B, int depth) const;
    void front_multiply(integer_t slo, integer_t shi, integer_t* upd, integer_t dim_upd,
			const DenseM_t& R, DenseM_t& Sr, DenseM_t& Sc) const;
    void extract_front(scalar_t* F11, scalar_t* F12, scalar_t* F21, integer_t dim_sep, integer_t dim_upd,
		       integer_t sep_begin, integer_t sep_end, integer_t* upd, int depth);
    void extract_F11_block(scalar_t* F, integer_t ldF, integer_t row, integer_t nr_rows, integer_t col, integer_t nr_cols);
    void extract_F12_block(scalar_t* F, integer_t ldF, integer_t row, integer_t nr_rows, integer_t col, integer_t nr_cols, integer_t* upd);
    void extract_F21_block(scalar_t* F, integer_t ldF, integer_t row, integer_t nr_rows, integer_t col, integer_t nr_cols, integer_t* upd);
    void apply_scaling(std::vector<scalar_t>& Dr, std::vector<scalar_t>& Dc);
    void apply_column_permutation(std::vector<integer_t>& perm);
    int read_matrix_market(std::string filename);
    real_t max_scaled_residual(scalar_t* x, scalar_t* b);
    void strumpack_mc64(int_t job, int_t* num, integer_t* perm, int_t liw, int_t* iw, int_t ldw,
			double* dw, int_t* icntl, int_t* info);
    void print_dense(std::string name);

    void extract_separator_2d(integer_t separator_end, const std::vector<std::size_t>& I, const std::vector<std::size_t>& J,
			      DistM_t& mat, MPI_Comm comm) const;
    void front_multiply_2d(integer_t sep_begin, integer_t sep_end, integer_t* upd, integer_t dim_upd,
			   DistM_t& R, DistM_t& Srow, DistM_t& Scol, int ctxt_all, MPI_Comm R_comm, int depth);
  };

  template<typename scalar_t,typename integer_t>
  CSCMatrix<scalar_t,integer_t>::CSCMatrix() : CompressedSparseMatrix<scalar_t,integer_t>() {}

  template<typename scalar_t,typename integer_t>
  CSCMatrix<scalar_t,integer_t>::CSCMatrix
  (integer_t n, integer_t* ptr, integer_t* ind, scalar_t* values, bool symm_sparsity)
    : CompressedSparseMatrix<scalar_t,integer_t>(n, ptr, ind, values, symm_sparsity) {
  }

  template<typename scalar_t,typename integer_t>
  CSCMatrix<scalar_t,integer_t>::CSCMatrix(integer_t n, integer_t nnz)
    : CompressedSparseMatrix<scalar_t,integer_t>(n, nnz) {}

  template<typename scalar_t,typename integer_t>
  CSCMatrix<scalar_t,integer_t>::CSCMatrix(const CSCMatrix<scalar_t,integer_t>& A) {
    clone_data(A);
  }

  template<typename scalar_t,typename integer_t> CSCMatrix<scalar_t,integer_t>&
  CSCMatrix<scalar_t,integer_t>::operator=(const CSCMatrix<scalar_t,integer_t>& A) {
    clone_data(A);
    return *this;
  }

  template<typename scalar_t,typename integer_t> void
  CSCMatrix<scalar_t,integer_t>::print_dense(std::string name) {
    scalar_t* M = new scalar_t[this->_n * this->_n];
    std::fill(M, M+(this->_n*this->_n), scalar_t(0.));
    for (integer_t col=0; col<this->_n; col++)
      for (integer_t j=this->_ptr[col]; j<this->_ptr[col+1]; j++)
	M[this->_ind[j] + col*this->_n] = this->_val[j];
    std::cout << name << " = [";
    for (integer_t row=0; row<this->_n; row++) {
      for (integer_t col=0; col<this->_n; col++)
	std::cout << M[row + this->_n * col] << " ";
      std::cout << ";" << std::endl;
    }
    std::cout << "];" << std::endl;
  }

  template<typename scalar_t,typename integer_t> CSCMatrix<scalar_t,integer_t>*
  CSCMatrix<scalar_t,integer_t>::clone() {
    CSCMatrix<scalar_t,integer_t>* t = new CSCMatrix<scalar_t,integer_t>();
    t->clone_data(*this);
    return t;
  }

  template<typename scalar_t,typename integer_t> void
  CSCMatrix<scalar_t,integer_t>::spmv(scalar_t* x, scalar_t* y) {
    for (integer_t i=0; i<this->_n; i++) y[i] = scalar_t(0.);
    for (integer_t i=0; i<this->_n; i++)
      for (integer_t j=this->_ptr[i]; j<this->_ptr[i+1]; j++)
	y[this->_ind[j]] += this->_val[j] * x[i];
    STRUMPACK_FLOPS((is_complex<scalar_t>()?4:1)*static_cast<long long int>(2.*double(this->nnz())-double(this->size())));
  }

#if defined(__INTEL_MKL__)
  template<> void CSCMatrix<float>::spmv(float* x, float* y)
  { char trans = 'N'; mkl_cspblas_scscgemv(&no, &n, this->_val, this->_ptr, this->_ind, x, y);
    STRUMPACK_FLOPS(static_cast<long long int>(2.*double(this->nnz())-double(this->size()))); }
  template<> void CSCMatrix<double>::spmv(double* x, double* y)
  { char trans = 'N'; mkl_cspblas_dcscgemv(&no, &n, this->_val, this->_ptr, this->_ind, x, y);
    STRUMPACK_FLOPS(static_cast<long long int>(2.*double(this->nnz())-double(this->size()))); }
  template<> void CSCMatrix<c_float>::spmv(c_float* x, c_float* y)
  { char trans = 'N'; mkl_cspblas_ccscgemv(&no, &n, this->_val, this->_ptr, this->_ind, x, y);
    STRUMPACK_FLOPS(4*static_cast<long long int>(2.*double(this->nnz())-double(this->size()))); }
  template<> void CSCMatrix<c_double>::spmv(c_double* x, c_double* y)
  { char trans = 'N'; mkl_cspblas_zcscgemv(&no, &n, this->_val, this->_ptr, this->_ind, x, y);
    STRUMPACK_FLOPS(4*static_cast<long long int>(2.*double(this->nnz())-double(this->size()))); }
#endif

  template<typename scalar_t,typename integer_t> void
  CSCMatrix<scalar_t,integer_t>::omp_spmv(scalar_t* x, scalar_t* y) {
    spmv(x, y);
  }

  template<typename scalar_t,typename integer_t> void
  CSCMatrix<scalar_t,integer_t>::strumpack_mc64
  (int_t job, int_t* num, integer_t* perm, int_t liw, int_t* iw, int_t ldw,
   double* dw, int_t* icntl, int_t* info) {
    int_t n = this->_n;
    int_t nnz = this->_nnz;
    double* dval = NULL;
    if (std::is_same<double,scalar_t>())
      dval = reinterpret_cast<double*>(this->_val);
    else {
      dval = new double[nnz];
      for (int_t i=0; i<nnz; i++) dval[i] = static_cast<double>(std::real(this->_val[i]));
    }
    if (std::is_same<integer_t,int_t>()) {
#pragma omp parallel
      {
#pragma omp for
	for (int_t i=0; i<=n; i++) this->_ptr[i]++;
#pragma omp for
	for (int_t i=0; i<nnz; i++) this->_ind[i]++;
      }
      strumpack_mc64ad_(&job, &n, &nnz, reinterpret_cast<int_t*>(this->_ptr),
			reinterpret_cast<int_t*>(this->_ind), dval, num, reinterpret_cast<int_t*>(perm),
			&liw, iw, &ldw, dw, icntl, info);
#pragma omp parallel
      {
#pragma omp for
	for (int_t i=0; i<=n; i++) this->_ptr[i]--;
#pragma omp for
	for (int_t i=0; i<nnz; i++) this->_ind[i]--;
#pragma omp for
	for (int_t i=0; i<n; i++) perm[i]--;
      }
    } else {
      int_t* c_ptr = new int_t[n+1+nnz+n];
      int_t* r_ind = c_ptr + n + 1;
      int_t* permutation = r_ind + nnz;
#pragma omp parallel
      {
#pragma omp for
	for (int_t i=0; i<=n; i++) c_ptr[i] = this->_ptr[i] + 1;
#pragma omp for
	for (int_t i=0; i<nnz; i++) r_ind[i] = this->_ind[i] + 1;
      }
      strumpack_mc64ad_(&job, &n, &nnz, c_ptr, r_ind, dval, num, permutation, &liw, iw, &ldw, dw, icntl, info);
#pragma omp parallel for
      for (int_t i=0; i<n; i++) perm[i] = permutation[i] - 1;
      delete[] c_ptr;
    }
    if (!std::is_same<double,scalar_t>())
      delete[] dval;
  }

  template<typename scalar_t,typename integer_t> void
  CSCMatrix<scalar_t,integer_t>::extract_separator_2d
  (integer_t separator_end, const std::vector<std::size_t>& I, const std::vector<std::size_t>& J,
   DistM_t& mat, MPI_Comm comm) const {
    std::cout << "TODO CSRMatrix<scalar_t,integer_t>::extract_separator_2d" << std::endl;
  }

  template<typename scalar_t,typename integer_t> void
  CSCMatrix<scalar_t,integer_t>::front_multiply_2d
  (integer_t sep_begin, integer_t sep_end, integer_t* upd, integer_t dim_upd,
   DistM_t& R, DistM_t& Srow, DistM_t& Scol, int ctxt_all, MPI_Comm R_comm, int depth) {
    std::cout << "TODO CSRMatrix<scalar_t,integer_t>::front_multiply_2d" << std::endl;
  }

  template<typename scalar_t,typename integer_t> void
  CSCMatrix<scalar_t,integer_t>::extract_separator
  (integer_t separator_end, const std::vector<std::size_t>& I, const std::vector<std::size_t>& J,
   DenseM_t& B, int depth) const {
    auto m = I.size();
    auto n = J.size();
    if (m == 0 || n == 0) return;
    if (depth < params::task_recursion_cutoff_level) {
      std::size_t loop_tasks = std::max(params::num_threads / (depth+1), 1);
      std::size_t MB = std::max(n / loop_tasks, integer_t(1));
      for (integer_t task=0; task<std::ceil(n/float(MB)); task++) {
#pragma omp task default(shared) firstprivate(task) if(loop_tasks>1)
	{
	  for (std::size_t i=task*MB; i<std::min((task+1)*MB,n); i++) {
	    auto row_min = this->_ind[this->_ptr[J[i]]]; // indices sorted in increasing order
	    auto row_max = this->_ind[this->_ptr[J[i]+1]-1];
	    for (std::size_t k=0; k<m; k++) {
	      auto irow = I[k];
	      if (irow >= row_min && irow <= row_max) {
		auto a_pos = this->_ptr[J[i]];
		while (a_pos<(this->_ptr[J[i]+1]-1) && this->_ind[a_pos]<irow) a_pos++;
		if (this->_ind[a_pos] == irow && (J[i] < separator_end || irow < separator_end)) // mod
		  B(k,i) = this->_val[a_pos];
		else B(k,i) = scalar_t(0.);
	      } else B(k,i) = scalar_t(0.);
	    }
	  }
	}
      }
#pragma omp taskwait
    } else {
      for (std::size_t i=0; i<n; i++) {
	auto row_min = this->_ind[this->_ptr[J[i]]]; // indices sorted in increasing order
	auto row_max = this->_ind[this->_ptr[J[i] + 1] - 1];
	for (std::size_t k=0; k<m; k++) {
	  auto irow = I[k];
	  if (irow >= row_min && irow <= row_max) {
	    auto a_pos = this->_ptr[J[i]];
	    while (a_pos<(this->_ptr[J[i]+1]-1) && this->_ind[a_pos]<irow) a_pos++;
	    if (this->_ind[a_pos] == irow && (J[i] < separator_end || irow < separator_end)) // mod
	      B(k,i) = this->_val[a_pos];
	    else B(k,i) = scalar_t(0.);
	  } else B(k,i) = scalar_t(0.);
	}
      }
    }
  }

  // assume F11, F12 and F21 are set to zero
  template<typename scalar_t,typename integer_t> void
  CSCMatrix<scalar_t,integer_t>::extract_front
  (scalar_t* F11, scalar_t* F12, scalar_t* F21, integer_t dim_sep, integer_t dim_upd,
   integer_t sep_begin, integer_t sep_end, integer_t* upd, int depth) {
    bool tasked = depth<params::task_recursion_cutoff_level;
    if (tasked) {
      integer_t loop_tasks = std::max(params::num_threads / (depth+1), 1);
      integer_t Bsep = std::max(dim_sep / loop_tasks, integer_t(1));
      for (integer_t task=0; task<std::ceil(dim_sep/float(Bsep)); task++) {
#pragma omp task default(shared) firstprivate(task) if(loop_tasks>1)
	{
	  for (integer_t col=sep_begin+task*Bsep; col<sep_begin+std::min((task+1)*Bsep,dim_sep); col++) {
	    integer_t upd_ptr = 0;
	    for (integer_t j=this->_ptr[col]; j<this->_ptr[col+1]; j++) {
	      integer_t row = this->_ind[j];
	      if (row >= sep_begin) {
		if (row < sep_end) F11[row-sep_begin + (col-sep_begin)*dim_sep] = this->_val[j];
		else {
		  while (upd_ptr<dim_upd && upd[upd_ptr]<row) upd_ptr++;
		  if (upd_ptr == dim_upd) break;
		  if (upd[upd_ptr] == row) F21[upd_ptr + (col-sep_begin)*dim_upd] = this->_val[j];
		}
	      }
	    }
	  }
	}
	integer_t Bupd = std::max(dim_upd / loop_tasks, integer_t(1));
	for (integer_t task=0; task<std::ceil(dim_upd/float(Bupd)); task++) {
#pragma omp task default(shared) firstprivate(task)
	  {
	    for (integer_t i=task*Bupd; i<std::min((task+1)*Bupd,dim_upd); i++) {
	      integer_t col = upd[i];
	      for (integer_t j=this->_ptr[col]; j<this->_ptr[col+1]; j++) {
		integer_t row = this->_ind[j];
		if (row >= sep_begin) {
		  if (row < sep_end) F12[row-sep_begin + i*dim_sep] = this->_val[j];
		  else break;
		}
	      }
	    }
	  }
	}
      }
#pragma omp taskwait
    } else {
      for (integer_t col=sep_begin; col<sep_end; col++) { // separator columns
	integer_t upd_ptr = 0;
	for (integer_t j=this->_ptr[col]; j<this->_ptr[col+1]; j++) {
	  integer_t row = this->_ind[j];
	  if (row >= sep_begin) {
	    if (row < sep_end) F11[row-sep_begin + (col-sep_begin)*dim_sep] = this->_val[j];
	    else {
	      while (upd_ptr<dim_upd && upd[upd_ptr]<row) upd_ptr++;
	      if (upd_ptr == dim_upd) break;
	      if (upd[upd_ptr] == row) F21[upd_ptr + (col-sep_begin)*dim_upd] = this->_val[j];
	    }
	  }
	}
      }
      for (integer_t i=0; i<dim_upd; i++) { // remaining columns
	integer_t col = upd[i];
	for (integer_t j=this->_ptr[col]; j<this->_ptr[col+1]; j++) {
	  integer_t row = this->_ind[j];
	  if (row >= sep_begin) {
	    if (row < sep_end) F12[row-sep_begin + i*dim_sep] = this->_val[j];
	    else break;
	  }
	}
      }
    }
  }

  // TODO tasking??
  template<typename scalar_t,typename integer_t> void
  CSCMatrix<scalar_t,integer_t>::front_multiply
  (integer_t slo, integer_t shi, integer_t* upd, integer_t dupd,
   const DenseM_t& R, DenseM_t& Sr, DenseM_t& Sc) const {
    auto nbvec = R.cols();
    auto ds = shi - slo;
    for (auto col=slo; col<shi; col++) { // separator columns
      std::size_t upd_ptr = 0;
      for (auto j=this->_ptr[col]; j<this->_ptr[col+1]; j++) {
	auto row = this->_ind[j];
	if (row >= slo) {
	  if (row < shi) {
	    for (std::size_t c=0; c<nbvec; c++) {
	      Sr(row-slo, c) += this->_val[j] * R(col-slo, c);
	      Sc(col-slo, c) += this->_val[j] * R(row-slo, c);
	    }
	    STRUMPACK_FLOPS((is_complex<scalar_t>()?4:1)*static_cast<long long int>(2.*double(nbvec)));
	  } else {
	    while (upd_ptr<dupd && upd[upd_ptr]<row) upd_ptr++;
	    if (upd_ptr == dupd) break;
	    if (upd[upd_ptr] == row) {
	      for (std::size_t c=0; c<nbvec; c++) {
		Sr(ds+upd_ptr, c) += this->_val[j] * R(col-slo, c);
		Sc(col-slo, c) += this->_val[j] * R(ds+upd_ptr, c);
	      }
	      STRUMPACK_FLOPS((is_complex<scalar_t>()?4:1)*static_cast<long long int>(2.*double(nbvec)));
	    }
	  }
	}
      }
    }
    for (integer_t i=0; i<dupd; i++) { // remaining columns
      auto col = upd[i];
      for (auto j=this->_ptr[col]; j<this->_ptr[col+1]; j++) {
	auto row = this->_ind[j];
	if (row >= slo) {
	  if (row < shi) {
	    for (std::size_t c=0; c<nbvec; c++) {
	      Sr(row-slo, c) += this->_val[j] * R(ds+i, c);
	      Sc(ds+i, c) += this->_val[j] * R(row-slo, c);
	    }
	    STRUMPACK_FLOPS((is_complex<scalar_t>()?4:1)*static_cast<long long int>(2.*double(nbvec)));
	  } else break;
	}
      }
    }
  }

  template<typename scalar_t,typename integer_t> void
  CSCMatrix<scalar_t,integer_t>::apply_scaling
  (std::vector<scalar_t>& Dr, std::vector<scalar_t>& Dc) {
#pragma omp parallel for
    for (integer_t j=0; j<this->_n; j++)
      for (integer_t i=this->_ptr[j]; i<this->_ptr[j+1]; i++)
	this->_val[i] = this->_val[i] * Dr[this->_ind[i]] * Dc[j];
    STRUMPACK_FLOPS((is_complex<scalar_t>()?6:1)*static_cast<long long int>(2.0*double(this->nnz())));
  }

  template<typename scalar_t,typename integer_t> void
  CSCMatrix<scalar_t,integer_t>::apply_column_permutation(std::vector<integer_t>& perm) {
    integer_t* new_ind = new integer_t[this->_nnz];
    integer_t* new_ptr = new integer_t[this->_n+1];
    scalar_t* new_val = new scalar_t[this->_nnz];
    integer_t nz = 0;
    for (integer_t c=0; c<this->_n; c++) {
      new_ptr[c] = nz;
      for (integer_t r=this->_ptr[perm[c]]; r<this->_ptr[perm[c]+1]; r++) {
	new_ind[nz] = this->_ind[r];
	new_val[nz++] = this->_val[r];
      }
    }
    new_ptr[this->_n] = nz;
    this->set_ind(new_ind);
    this->set_val(new_val);
    this->set_ptr(new_ptr);
  }

  template<typename scalar_t,typename integer_t> int
  CSCMatrix<scalar_t,integer_t>::read_matrix_market(std::string filename) {
    std::vector<std::tuple<integer_t,integer_t,scalar_t>> A;
    try {
      A = this->read_matrix_market_entries(filename);
    } catch (...) { return 1; }
    std::sort(A.begin(), A.end(), [](const std::tuple<integer_t,integer_t,scalar_t>& a, const std::tuple<integer_t,integer_t,scalar_t>& b) -> bool {
	// sort based on the column,row indices
	return std::make_tuple(std::get<1>(a),std::get<0>(a)) < std::make_tuple(std::get<1>(b), std::get<0>(b));
      });

    this->_ptr = new integer_t[this->_n+1];
    this->_ind = new integer_t[this->_nnz];
    this->_val = new scalar_t[this->_nnz];
    integer_t row = -1;
    for (integer_t i=0; i<this->_nnz; i++) {
      this->_val[i] = std::get<2>(A[i]);
      this->_ind[i] = std::get<0>(A[i]);
      auto new_row = std::get<1>(A[i]);
      if (new_row != row) {
	for (int j=row+1; j<=new_row; j++) this->_ptr[j] = i;
	row = new_row;
      }
    }
    for (int j=row+1; j<=this->_n; j++) this->_ptr[j] = this->_nnz;
    return 0;
  }

  template<typename scalar_t,typename integer_t> void
  CSCMatrix<scalar_t,integer_t>::extract_F11_block
  (scalar_t* F, integer_t ldF, integer_t row, integer_t nr_rows,
   integer_t col, integer_t nr_cols) {
    std::cout << "CSCMatrix<scalar_t,integer_t>::extract_F11 not implemented yet!" << std::endl;
    abort();
  }

  template<typename scalar_t,typename integer_t> void
  CSCMatrix<scalar_t,integer_t>::extract_F12_block
  (scalar_t* F, integer_t ldF, integer_t row, integer_t nr_rows,
   integer_t col, integer_t nr_cols, integer_t* upd) {
    std::cout << "CSCMatrix<scalar_t,integer_t>::extract_F12 not implemented yet!" << std::endl;
    abort();
  }

  template<typename scalar_t,typename integer_t> void
  CSCMatrix<scalar_t,integer_t>::extract_F21_block
  (scalar_t* F, integer_t ldF, integer_t row, integer_t nr_rows,
   integer_t col, integer_t nr_cols, integer_t* upd) {
    std::cout << "CSCMatrix<scalar_t,integer_t>::extract_F21 not implemented yet!" << std::endl;
    abort();
  }

  template<typename scalar_t,typename integer_t> real_t
  CSCMatrix<scalar_t,integer_t>::max_scaled_residual(scalar_t* x, scalar_t* b) {
    scalar_t* true_res = new scalar_t[this->_n];
    scalar_t* abs_res = new scalar_t[this->_n];

    for (integer_t i=0; i<this->_n; i++) true_res[i] = b[i];
    for (integer_t i=0; i<this->_n; i++)
      for (integer_t j=this->_ptr[i]; j<this->_ptr[i+1]; j++)
	true_res[this->_ind[j]] -= this->_val[j] * x[i];

    for (integer_t i=0; i<this->_n; i++) abs_res[i] = std::abs(b[i]);
    for (integer_t i=0; i<this->_n; i++)
      for (integer_t j=this->_ptr[i]; j<this->_ptr[i+1]; j++)
	abs_res[this->_ind[j]] += std::abs(this->_val[j]) * std::abs(x[i]);

    real_t m = 0.;
    for (integer_t i=0; i<this->_n; i++)
      m = std::max(std::abs(m), std::abs(true_res[i]) / std::abs(abs_res[i]));

    delete[] true_res;
    delete[] abs_res;
    return m;
  }

} // end namespace strumpack

#endif //CSCMATRIX_H
