/*
 * STRUMPACK -- STRUctured Matrices PACKage, Copyright (c) 2014, The
 * Regents of the University of California, through Lawrence Berkeley
 * National Laboratory (subject to receipt of any required approvals
 * from the U.S. Dept. of Energy).  All rights reserved.
 *
 * If you have questions about your rights to use or distribute this
 * software, please contact Berkeley Lab's Technology Transfer
 * Department at TTD@lbl.gov.
 *
 * NOTICE. This software is owned by the U.S. Department of Energy. As
 * such, the U.S. Government has been granted for itself and others
 * acting on its behalf a paid-up, nonexclusive, irrevocable,
 * worldwide license in the Software to reproduce, prepare derivative
 * works, and perform publicly and display publicly.  Beginning five
 * (5) years after the date permission to assert copyright is obtained
 * from the U.S. Department of Energy, and subject to any subsequent
 * five (5) year renewals, the U.S. Government is granted for itself
 * and others acting on its behalf a paid-up, nonexclusive,
 * irrevocable, worldwide license in the Software to reproduce,
 * prepare derivative works, distribute copies to the public, perform
 * publicly and display publicly, and to permit others to do so.
 *
 * Developers: Pieter Ghysels, Francois-Henry Rouet, Xiaoye S. Li.
 *             (Lawrence Berkeley National Lab, Computational Research
 *             Division).
 */
#ifndef HSS_MATRIX_SKETCH_HPP
#define HSS_MATRIX_SKETCH_HPP

#include "misc/RandomWrapper.hpp"
#include "misc/Tools.hpp"
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace strumpack {
  namespace HSS {

    template<typename scalar_t> class BinaryCRSMarix {
    public:
      BinaryCRSMarix(std::size_t n_rows, std::size_t n_cols) :
        nnz_(std::size_t(0)), n_cols_(n_cols), n_rows_(n_rows),
        one_(scalar_t(1.)), col_ind_({}), row_ptr_({std::size_t(0)}) {}
      BinaryCRSMarix(std::vector<std::size_t> col_ind,
                     std::vector<std::size_t> row_ptr,
                     std::size_t n_cols) :
        nnz_(col_ind.size()), n_cols_(n_cols),
        n_rows_(row_ptr.size() - 1), one_(scalar_t(1.)),
        col_ind_(col_ind), row_ptr_(row_ptr) {
      }

      void print() {
        std::cout << "row ptr: ";
        for (std::size_t i = 0; i < row_ptr_.size(); i++)
          std::cout << row_ptr_[i] << " ";
        std::cout << std::endl << "col ind: ";
        for (std::size_t i = 0; i < col_ind_.size(); i++)
          std::cout << col_ind_[i] << " ";
        std::cout << std::endl;
        std::cout << "val: " << one_ << std::endl;
        std::cout << "n_cols " << n_cols_ << std::endl;
      }

      void print_as_dense() {
        for (std::size_t i=0; i<row_ptr_.size()-1; i++) {
          std::size_t start = row_ptr_[i], end = row_ptr_[i+1];
          for (std::size_t j=0; j<n_cols_; j++)
            if (std::find(col_ind_.begin() + start,
                          col_ind_.begin() + end, j) !=
                col_ind_.begin() + end)
              std::cout << one_ << " ";
            else
              std::cout << "0 ";
          std::cout << std::endl;
        }
      }

      void add_row(std::vector<std::size_t> new_col_ind_) {
        std::size_t added_nnz_ = new_col_ind_.size();
        // append col_inds
        col_ind_.insert(std::end(col_ind_),
                        std::begin(new_col_ind_),
                        std::end(new_col_ind_));
        // update nnz_
        nnz_ += added_nnz_;
        // update row_ptr
        row_ptr_.push_back(nnz_);
        // update n_rows
        n_rows_ += 1;
      }

      /**
       * Appends the cols of a second B-CRS
       * matrix to the end of this matrix:
       */
      void append_cols(BinaryCRSMarix<scalar_t>& T) {
        if (T.n_rows() != n_rows_) {
          std::cout << "# Cannot append a matrix with"
                    << " the wrong number of rows" << std::endl
                    << "# original rows: " << n_rows_
                    << "  new rows: " << T.n_rows() << std::endl;
          return;
        }
        const auto rows_T = T.get_row_ptr();
        const auto col_T = T.get_col_inds();
        std::vector<std::size_t> new_row_ptr_;
        new_row_ptr_.reserve(rows_T.size());
        new_row_ptr_.push_back(std::size_t(0));
        std::vector<std::size_t> new_col_inds;
        new_col_inds.reserve(col_T.size()+col_ind_.size());
        // update col indices
        for (std::size_t i=0; i<row_ptr_.size()-1; i++) {
          for (std::size_t j=row_ptr_[i]; j<row_ptr_[i+1]; j++)
            new_col_inds.push_back(col_ind_[j]);
          for (std::size_t j=rows_T[i]; j<rows_T[i+1]; j++)
            new_col_inds.push_back(col_T[j] + n_cols());
          new_row_ptr_.push_back(row_ptr_[i+1]+rows_T[i+1]);
        }
        nnz_ += T.nnz();
        n_cols_ += T.n_cols();
        col_ind_ = new_col_inds;
        row_ptr_ = new_row_ptr_;
      }

      void set_nnz_value(scalar_t one) { one_ = one; }
      scalar_t nnz_value() { return one_; }
      std::size_t nnz() { return nnz_; }
      std::size_t n_rows() { return n_rows_; }
      std::size_t n_cols() { return n_cols_; }

      const std::vector<std::size_t>& get_row_ptr() const {
        return row_ptr_;
      }
      const std::vector<std::size_t>& get_col_inds() const {
        return col_ind_;
      }

      void set_ptrs(std::vector<std::size_t> col_ind,
                    std::vector<std::size_t> row_ptr) {
        col_ind_ = std::move(col_ind);
        nnz_ = col_ind_.size();
        row_ptr_ = std::move(row_ptr);
        n_rows_ = row_ptr_.size() - 1;
      }

    private:
      std::size_t nnz_, n_cols_, n_rows_;
      scalar_t one_;
      std::vector<std::size_t> col_ind_, row_ptr_;
    };

    template<typename scalar_t> class BinaryCCSMarix {
    public:
      BinaryCCSMarix(std::size_t n_rows, std::size_t n_cols) :
        nnz_(std::size_t(0)), n_cols_(n_cols), n_rows_(n_rows),
        one_(scalar_t(1.)), row_ind_({}), col_ptr_({std::size_t(0)}) {}
      BinaryCCSMarix(std::vector<std::size_t> row_ind,
                     std::vector<std::size_t> col_ptr,
                     std::size_t n_rows) :
        nnz_(row_ind.size()), n_cols_(col_ptr.size()-1),
        n_rows_(n_rows), one_(scalar_t(1.)),
        row_ind_(row_ind), col_ptr_(col_ptr) {}

      void print() {
        std::cout << "col ptr: ";
        for (std::size_t i=0; i<col_ptr_.size(); i++)
          std::cout << col_ptr_[i] << " ";
        std::cout << std::endl << "row ind: ";
        for (std::size_t i=0; i<row_ind_.size(); i++)
          std::cout << row_ind_[i] << " ";
        std::cout << std::endl << "val: " << one_ << " " << std::endl;
      }

      void print_as_dense() {
        std::vector<std::string> vec(n_rows(), "");
        for (std::size_t i=0; i<col_ptr_.size()-1; i++) {
          std::size_t start = col_ptr_[i], end = col_ptr_[i+1];
          for (std::size_t j=0; j<n_rows(); j++)
            if (std::find(row_ind_.begin() + start,
                          row_ind_.begin() + end, j) !=
                row_ind_.begin() + end)
              vec[j] += std::to_string(one_) + " ";
            else
              vec[j] += "0 ";
        }
        for (std::size_t i=0; i<n_rows(); i++)
          std::cout << vec[i] << std::endl;
      }

      void append_cols(BinaryCCSMarix<scalar_t>& T) {
        if (T.n_rows() != n_rows_) {
          std::cout << "# Cannot append a matrix with"
                    << " the wrong number of rows" << std::endl;
          return;
        }
        const auto new_cols = T.get_col_ptr();
        const auto new_row_inds = T.get_row_inds();
        row_ind_.reserve(row_ind_.size()+new_row_inds.size());
        row_ind_.insert(row_ind_.end(),
                        new_row_inds.begin(),
                        new_row_inds.end());
        // add new columns:
        col_ptr_.reserve(col_ptr_.size()+new_cols.size());
        for (size_t i=1; i<new_cols.size(); i++)
          col_ptr_.push_back(new_cols[i] + nnz_);
        // one and n_rows_ does not change
        nnz_ += T.nnz();
        n_cols_ += T.n_cols();
      }

      void add_col(std::vector<std::size_t> new_row_ind_) {
        std::size_t added_nnz_ = new_row_ind_.size();
        // append col_inds
        row_ind_.insert(std::end(row_ind_),
                        std::begin(new_row_ind_),
                        std::end(new_row_ind_));
        // update nnz_
        nnz_ += added_nnz_;
        // update row_ptr
        col_ptr_.push_back(nnz_);
        // update n_rows
        n_cols_ += 1;
      }

      void set_nnz_value(scalar_t one) { one_ = one; }
      scalar_t nnz_value() { return one_; }
      std::size_t nnz() { return nnz_; }
      std::size_t n_cols() { return n_cols_; }
      std::size_t n_rows() { return n_rows_; }

      const  std::vector<std::size_t>& get_col_ptr() const {
        return col_ptr_;
      }
      const  std::vector<std::size_t>& get_row_inds() const {
        return row_ind_;
      }

      void set_ptrs(std::vector<std::size_t> row_ind,
                    std::vector<std::size_t> col_ptr) {
        row_ind_ = std::move(row_ind);
        nnz_ = row_ind_.size();
        col_ptr_ = std::move(col_ptr);
        n_cols_ = col_ptr_.size() - 1;
        // update nnz + n_rows
      }

    private:
      std::size_t nnz_, n_cols_, n_rows_;
      scalar_t one_;
      std::vector<std::size_t> row_ind_, col_ptr_;
    };

    template<typename scalar_t, typename integer_t>
    class SJLTGenerator {
    public:
      SJLTGenerator() {
        seed_ = std::chrono::system_clock::now().
          time_since_epoch().count();
        e_.seed(seed_);
      }
      SJLTGenerator(integer_t seed) {
        seed_ = seed;
        e_.seed(seed_);
      }
      void set_seed(integer_t seed) {
        seed_ = seed;
        e_.seed(seed_);
      }
      void createSJLTCRS(BinaryCRSMarix<scalar_t>& A,
                         BinaryCRSMarix<scalar_t>& B,
                         BinaryCCSMarix<scalar_t>& Ac,
                         BinaryCCSMarix<scalar_t>& Bc,
                         std::size_t nnz, std::size_t n_rows,
                         std::size_t n_cols) {
        if (nnz > n_cols) {
          std::cout << "# POSSIBLE ERROR: nnz bigger than n_cols"
                    << std::endl
                    << "# setting nnz to n_cols" << std::endl;
          nnz = n_cols;
        }
        // set the nnz value for each of the matrices:
        A.set_nnz_value(scalar_t(1));
        Ac.set_nnz_value(scalar_t(1));
        B.set_nnz_value(scalar_t(-1));
        Bc.set_nnz_value(scalar_t(-1));
        // We'll be generating 8 pointers for the 4 matrices:
        // rowise:
        std::vector<std::size_t> A_row_ptr(1+n_rows,0);
        std::vector<std::size_t> A_col_inds;
        A_col_inds.reserve(n_rows * nnz);
        std::vector<std::size_t> B_row_ptr(1+n_rows, 0);
        std::vector<std::size_t> B_col_inds;
        B_col_inds.reserve(n_rows * nnz);
        // columnwise:
        std::vector<std::size_t> Ac_col_ptr(1+n_cols, 0);
        std::vector<std::vector<std::size_t>>
          Ac_row_inds(n_cols, std::vector<std::size_t>());
        for (auto v : Ac_row_inds)
          v.reserve(std::size_t(nnz*n_rows/n_cols+1));
        std::vector<std::size_t> Bc_col_ptr(1+n_cols, 0);
        std::vector<std::vector<std::size_t>>
          Bc_row_inds(n_cols, std::vector<std::size_t>());
        for (auto v : Bc_row_inds)
          v.reserve(std::size_t(nnz*n_rows/n_cols+1));
        // SJLT generation algorithm:
        std::vector<std::size_t> col_inds;
        col_inds.reserve(n_cols);
        for (std::size_t j=0; j<n_cols; j++)
          col_inds.push_back(j);
        std::vector<int> nums = { 1,-1 };
        std::size_t a_nnz = 0, b_nnz = 0;
        for (std::size_t i=0; i<n_rows; i++) {
          // sample nnz column indices
          std::shuffle(col_inds.begin(), col_inds.end(), e_);
          a_nnz = 0, b_nnz = 0;
          for (std::size_t j=0; j<nnz; j++) {
            // decide whether each is +- 1
            std::shuffle(nums.begin(), nums.end(), e_);
            if (nums[0] == 1) {
              // belongs to A
              A_col_inds.push_back(col_inds[j]);
              a_nnz++;
              // CCS processing:
              Ac_col_ptr[col_inds[j]+1]++;
              Ac_row_inds[col_inds[j]].push_back(i);
            } else {
              // belongs to B
              B_col_inds.push_back(col_inds[j]);
              b_nnz++;
              // CCS processing:
              Bc_col_ptr[col_inds[j]+1]++;
              Bc_row_inds[col_inds[j]].push_back(i);
            }
          }
          // put in A and B row into A and B
          A_row_ptr[i+1] = A_row_ptr[i] + a_nnz;
          B_row_ptr[i+1] = B_row_ptr[i] + b_nnz;
        }
        A.set_ptrs(A_col_inds,A_row_ptr);
        B.set_ptrs(B_col_inds, B_row_ptr);
        // Columnwise processing:
        // update col_ptr by summing previous indices:
        for (std::size_t i=1; i<Ac_col_ptr.size(); i++)
          Ac_col_ptr[i] += Ac_col_ptr[i-1];
        for (std::size_t i=1; i<Bc_col_ptr.size(); i++)
          Bc_col_ptr[i] += Bc_col_ptr[i-1];
        // update row_inds by unravelling vectors:
        std::vector<std::size_t> Ac_final_inds;
        Ac_final_inds.reserve(Ac_col_ptr[Ac_col_ptr.size()-1]);
        for (auto&& v : Ac_row_inds)
          Ac_final_inds.insert(Ac_final_inds.end(), v.begin(), v.end());
        std::vector<std::size_t> Bc_final_inds;
        Bc_final_inds.reserve(Bc_col_ptr[Ac_col_ptr.size()-1]);
        for (auto&& v : Bc_row_inds)
          Bc_final_inds.insert(Bc_final_inds.end(), v.begin(), v.end());
        Ac.set_ptrs(Ac_final_inds,Ac_col_ptr);
        Bc.set_ptrs(Bc_final_inds, Bc_col_ptr);
      }

      void createSJLTCRS_Chunks(BinaryCRSMarix<scalar_t>& A,
                                BinaryCRSMarix<scalar_t>& B,
                                BinaryCCSMarix<scalar_t>& Ac,
                                BinaryCCSMarix<scalar_t>& Bc,
                                std::size_t nnz, std::size_t n_rows,
                                std::size_t n_cols) {
        if (nnz > n_cols) {
          std::cout << "# POSSIBLE ERROR: nnz bigger than n_cols"
                    << std::endl
                    << "# setting nnz to n_cols" << std::endl;
          nnz = n_cols;
        }
        // set the nnz value for each of the matrices:
        A.set_nnz_value(scalar_t(1));
        Ac.set_nnz_value(scalar_t(1));
        B.set_nnz_value(scalar_t(-1));
        Bc.set_nnz_value(scalar_t(-1));
        // We'll be generating 8 pointers for the 4 matrices:
        // rowise:
        std::vector<std::size_t> A_row_ptr(1+n_rows,0);
        std::vector<std::size_t> A_col_inds;
        A_col_inds.reserve(n_rows*nnz);
        std::vector<std::size_t> B_row_ptr(1+n_rows, 0);
        std::vector<std::size_t> B_col_inds;
        B_col_inds.reserve(n_rows*nnz);
        // columnwise:
        std::vector<std::size_t> Ac_col_ptr(1+n_cols, 0);
        std::vector<std::vector<std::size_t>>
          Ac_row_inds(n_cols, std::vector<std::size_t>());
        for (auto v : Ac_row_inds)
          v.reserve(std::size_t(nnz*n_rows/n_cols+1));
        std::vector<std::size_t> Bc_col_ptr(1+n_cols, 0);
        std::vector<std::vector<std::size_t>>
          Bc_row_inds(n_cols, std::vector<std::size_t>());
        for (auto v : Bc_row_inds)
          v.reserve(std::size_t(nnz*n_rows / n_cols+1));
        // SJLT generation algorithm:
        if (nnz != 0) {
          std::size_t chunk_size = n_cols/nnz;
          std::uniform_int_distribution<> shift(0, int(chunk_size)-1);
          std::uniform_int_distribution<> sign(0, 1);
          std::size_t a_nnz = 0, b_nnz = 0;
          for (std::size_t i=0; i<n_rows; i++) {
            a_nnz = 0, b_nnz = 0;
            for (std::size_t j=0; j<nnz; j++) {
              std::size_t index = shift(e_) + chunk_size*j;
              // decide whether each is +- 1
              if (sign(e_) == 0) {
                // belongs to A
                A_col_inds.push_back(index);
                a_nnz++;
                // CCS processing:
                Ac_col_ptr[index+1]++;
                Ac_row_inds[index].push_back(i);
              } else {
                // belongs to B
                B_col_inds.push_back(index);
                b_nnz++;
                // CCS processing:
                Bc_col_ptr[index+1]++;
                Bc_row_inds[index].push_back(i);
              }
            }
            // put in A and B row into A and B
            A_row_ptr[i+1] = A_row_ptr[i] + a_nnz;
            B_row_ptr[i+1] = B_row_ptr[i] + b_nnz;
          }
        }
        A.set_ptrs(A_col_inds, A_row_ptr);
        B.set_ptrs(B_col_inds, B_row_ptr);
        // Columnwise processing:
        // update col_ptr by summing previous indices:
        for (std::size_t i=1; i<Ac_col_ptr.size(); i++)
          Ac_col_ptr[i] += Ac_col_ptr[i-1];
        for (std::size_t i=1; i<Bc_col_ptr.size(); i++)
          Bc_col_ptr[i] += Bc_col_ptr[i-1];
        // update row_inds by unravelling vectors:
        std::vector<std::size_t> Ac_final_inds;
        Ac_final_inds.reserve(Ac_col_ptr[Ac_col_ptr.size()-1]);
        for (auto&& v : Ac_row_inds)
          Ac_final_inds.insert(Ac_final_inds.end(), v.begin(), v.end());
        std::vector<std::size_t> Bc_final_inds;
        Bc_final_inds.reserve(Bc_col_ptr[Ac_col_ptr.size()-1]);
        for (auto&& v : Bc_row_inds)
          Bc_final_inds.insert(Bc_final_inds.end(), v.begin(), v.end());
        Ac.set_ptrs(Ac_final_inds,Ac_col_ptr);
        Bc.set_ptrs(Bc_final_inds, Bc_col_ptr);
      }

      void SJLTDenseSketch(DenseMatrix<scalar_t>& B, std::size_t nnz) {
        if (nnz > B.cols()) {
          std::cout << "# error nnz too large" << std::endl
                    << "# n_cols = " << B.cols() << std::endl
                    << "# nnz = " << nnz << std::endl;
          return; // either make error or make nnz - B.cols()
        }
        // set initial B to zero:
        B.zero();
        std::vector<int> col_inds;
        for (unsigned int j=0; j<B.cols(); j++)
          col_inds.push_back(j);
        std::vector<scalar_t> nums = {scalar_t(1.), scalar_t(-1.)};
        for (std::size_t i=0; i<B.rows(); i++) {
          // sample nnz column indices breaks in second loop here
          // take the first nnz elements nonzero, else 0
          std::shuffle(col_inds.begin(), col_inds.end(), e_);
          for (std::size_t j=0; j<nnz; j++) {
            // decide whether each is +- 1
            std::shuffle(nums.begin(), nums.end(), e_);
            B(i, col_inds[j]) = nums[0];
          }
        }
      }

    private:
      integer_t seed_;
      std::default_random_engine e_;
    };


    /*
     * SJLT matrix S = (1/sqrt(nnz))(A - B)
     */
    template<typename scalar_t, typename integer_t>
    class SJLTMatrix {
    public:
      SJLTMatrix(SJLTGenerator<scalar_t, integer_t>& g, std::size_t nnz,
                 std::size_t n_rows, std::size_t n_cols, bool chunk) :
        g_(&g), nnz_(nnz), n_rows_(n_rows), n_cols_(n_cols),
        A_(BinaryCRSMarix<scalar_t>(0, n_cols)),
        B_(BinaryCRSMarix<scalar_t>(0, n_cols)),
        Ac_(BinaryCCSMarix<scalar_t>(n_rows, 0)),
        Bc_(BinaryCCSMarix<scalar_t>(n_rows, 0)),
        chunk_(chunk) {
        if (chunk_)
          g_->createSJLTCRS_Chunks(A_, B_, Ac_, Bc_, nnz_, n_rows_, n_cols_);
        else
          g_->createSJLTCRS(A_, B_, Ac_, Bc_, nnz_, n_rows_, n_cols_);
      }

      void add_columns(std::size_t new_cols, std::size_t nnz) {
        if (nnz > new_cols) {
          std::cout << "# nnz bigger than n_cols cannot proceed" << std::endl;;
          return;
        }
        BinaryCRSMarix<scalar_t> A_temp(0, new_cols);
        BinaryCRSMarix<scalar_t> B_temp(0, new_cols);
        /* Fix this */
        BinaryCCSMarix<scalar_t> Ac_temp(n_rows_, 0);
        BinaryCCSMarix<scalar_t> Bc_temp(n_rows_, 0);
        if (chunk_)
          g_->createSJLTCRS_Chunks(A_temp, B_temp, Ac_temp, Bc_temp,
                                   nnz, n_rows_, new_cols);
        else
          g_->createSJLTCRS(A_temp, B_temp, Ac_temp, Bc_temp,
                            nnz, n_rows_, new_cols);
        A_.append_cols(A_temp);
        B_.append_cols(B_temp);
        Ac_.append_cols(Ac_temp);
        Bc_.append_cols(Bc_temp);
        n_cols_ += new_cols;
        nnz_ += nnz;
      }

      void append_sjlt_matrix(SJLTMatrix<scalar_t,integer_t>& temp) {
        if (temp.get_n_rows() != n_rows_)
          std::cout << "# wrong shape to append" << std::endl;;
        nnz_ += temp.get_nnz();
        n_cols_ += temp.get_n_cols();
        A_.append_cols(temp.get_A());
        B_.append_cols(temp.get_B());
        Ac_.append_cols(temp.get_Ac());
        Bc_.append_cols(temp.get_Bc());
      }

      void print_sjlt_as_dense() {
        const auto rows_A = A_.get_row_ptr();
        const auto col_A = A_.get_col_inds();
        const auto rows_B = B_.get_row_ptr();
        const auto col_B = B_.get_col_inds();
        for (std::size_t i=0; i<n_rows_; i++) {
          std::size_t startA = rows_A[i], endA = rows_A[i+1];
          std::size_t startB = rows_B[i], endB = rows_B[i+1];
          for (std::size_t j=0; j<n_cols_; j++) {
            if (std::find(col_A.begin()+startA,
                          col_A.begin()+endA, j) != col_A.begin()+endA)
              std::cout << "1 ";
            else if (std::find(col_B.begin()+startB,
                               col_B.begin()+endB, j) !=
                     col_B.begin()+endB)
              std::cout << "-1 ";
            else
              std::cout << "0 ";
          }
          std::cout << std::endl;
        }
      }

      BinaryCRSMarix<scalar_t>& get_A() { return A_; }
      BinaryCRSMarix<scalar_t>& get_B() { return B_; }
      BinaryCCSMarix<scalar_t>& get_Ac() { return Ac_; }
      BinaryCCSMarix<scalar_t>& get_Bc() { return Bc_; }

      std::size_t get_n_rows() const { return n_rows_; }
      std::size_t get_n_cols() const { return n_cols_; }
      std::size_t get_nnz() const { return nnz_; }
      SJLTGenerator<scalar_t,integer_t> & get_g() { return *g_; }

      bool get_chunk(){ return chunk_; }

      // convert SJLT class to densematrix
      DenseMatrix<scalar_t> SJLT_to_dense() {
        DenseMatrix<scalar_t> S(n_rows_, n_cols_);
        const auto rows_A = A_.get_row_ptr();
        const auto col_A = A_.get_col_inds();
        const auto rows_B = B_.get_row_ptr();
        const auto col_B = B_.get_col_inds();
        for (std::size_t i=0; i<n_rows_; i++) {
          std::size_t startA = rows_A[i], endA = rows_A[i+1];
          std::size_t startB = rows_B[i], endB = rows_B[i+1];
          for (std::size_t j=0; j<n_cols_; j++) {
            if (std::find(col_A.begin()+startA,
                          col_A.begin()+endA, j) !=
                col_A.begin() + endA)
              S(i, j) = 1;
            else if (std::find(col_B.begin()+startB,
                               col_B.begin()+endB, j) !=
                     col_B.begin()+endB)
              S(i, j) = -1;
            else
              S(i, j) = 0;
          }
        }
        return S;
      }

    private:
      SJLTGenerator<scalar_t,integer_t>* g_ = nullptr;
      std::size_t nnz_, n_rows_, n_cols_;
      BinaryCRSMarix<scalar_t> A_, B_;
      BinaryCCSMarix<scalar_t> Ac_, Bc_;
      bool chunk_ = true;
    };

    // multiplication A <- M*S(i:i+m,j:j+n)
    // where M is dense and S is sparse SJLT matrix
    template<typename scalar_t, typename integer_t> void
    matrix_times_SJLT_seq(const DenseMatrix<scalar_t>& M ,
                          SJLTMatrix<scalar_t, integer_t>& S,
                          DenseMatrix<scalar_t>& A,
                          scalar_t alpha, scalar_t beta,
                          std::size_t m, std::size_t n,
                          std::size_t i, std::size_t j) {
      // if the submatrix is 0x0 then we use the full S matrix
      m = m > 0 ? m : S.get_n_rows();
      n = n > 0 ? n : S.get_n_cols();
      //outer products method:
      if (beta == scalar_t(0.))
        A.zero();
      else if (beta != scalar_t(1.))
        A.scale(beta);
      const auto rows_A = S.get_A().get_row_ptr();
      const auto col_A = S.get_A().get_col_inds();
      const auto rows_B = S.get_B().get_row_ptr();
      const auto col_B = S.get_B().get_col_inds();
      std::size_t rows = M.rows();
      if (alpha == scalar_t(1.)) {
        for (size_t k=i; k<i+m; k++) {
          std::size_t start_A = rows_A[k], end_A = rows_A[k+1];
          std::size_t startB = rows_B[k], endB = rows_B[k+1];
          auto Mk = M.ptr(0,k-i);
          // add cols
          for (std::size_t l=start_A; l<end_A; l++) {
            auto cAl = col_A[l] - j;
            if (cAl < n && cAl >= 0)
              for (size_t r=0; r<rows; r++)
                A(r,cAl) += Mk[r]; // M(r,k);
          }
          // subtract cols
          for (std::size_t l=startB; l<endB; l++) {
            auto cBl = col_B[l] - j;
            if (cBl >= 0 && cBl < n)
              for (size_t r=0; r<rows; r++)
                A(r, cBl) -= Mk[r]; // M(r,k);
          }
        }
      } else if (alpha == scalar_t(-1.)) {
        for (size_t k=i; k<i+m; k++) {
          std::size_t start_A = rows_A[k], end_A = rows_A[k+1];
          std::size_t startB = rows_B[k], endB = rows_B[k+1];
          auto Mk = M.ptr(0, k-i);
          // add cols
          for (std::size_t l=start_A; l<end_A; l++) {
            auto cAl = col_A[l] - j;
            if (cAl < n && cAl >= 0)
              for (size_t r=0; r<rows; r++)
                A(r, cAl) -= Mk[r]; // M(r,k);
          }
          // subtract cols
          for (std::size_t l=startB; l<endB; l++) {
            auto cBl = col_B[l] - j;
            if(cBl >= 0 && cBl < n)
              for(size_t r = 0; r < rows; r++)
                A(r, cBl) += Mk[r]; // M(r,k);
          }
        }
      } else {
        for (size_t k=i; k<i+m; k++) {
          std::size_t start_A = rows_A[k], end_A = rows_A[k+1];
          std::size_t startB = rows_B[k], endB = rows_B[k+1];
          auto Mk = M.ptr(0, k-i);
          // add cols
          for (std::size_t l=start_A; l<end_A; l++) {
            auto cAl = col_A[l] - j;
            if (cAl < n && cAl >= 0)
              for (size_t r=0; r<rows; r++)
                A(r, cAl) += alpha * Mk[r]; // M(r,k);
          }
          // subtract cols
          for (std::size_t l=startB; l<endB; l++) {
            auto cBl = col_B[l] - j;
            if (cBl >= 0 && cBl < n)
              for (size_t r=0; r<rows; r++)
                A(r, cBl) -= alpha * Mk[r]; // M(r,k);
          }
        }
      }
    }

    // given M,S,m,n,i,j : A <- alpha * M *S(i:i+m,j:j+n) + beta * A
    template<typename scalar_t, typename integer_t> void
    matrix_times_SJLT(const DenseMatrix<scalar_t>& M ,
                      SJLTMatrix<scalar_t, integer_t>& S,
                      DenseMatrix<scalar_t>& A,
                      std::size_t m = 0 , std::size_t n=  0,
                      std::size_t i = 0, std::size_t j = 0,
                      scalar_t alpha = 1., scalar_t beta = 0.) {
#if defined(_OPENMP)
      int rows = M.rows();
      int T = omp_get_max_threads();
      int B = rows / T;
#pragma omp parallel for schedule(static,1)
      for (int r=0; r<rows; r+=B) {
        DenseMatrixWrapper<scalar_t> Asub
          (std::min(rows-r, B), A.cols(), A, r, 0);
        auto Msub = ConstDenseMatrixWrapperPtr<scalar_t>
          (std::min(rows-r, B), M.cols(), M, r, 0);
        matrix_times_SJLT_seq(*Msub, S, Asub, alpha, beta, m,n,i,j);
      }
#else
      matrix_times_SJLT_seq(M, S, A, alpha, beta, m,n,i,j);
#endif
    }

    /**
     * Given M,S,m,n,i,j : A <- alpha * M^* S(i:i+m,j:j+n) + beta * A
     * using inner products of columns of M and S
     */
    template<typename scalar_t, typename integer_t> void
    matrixT_times_SJLT(const DenseMatrix<scalar_t>& M ,
                       SJLTMatrix<scalar_t, integer_t>& S,
                       DenseMatrix<scalar_t>& A,
                       std::size_t m = 0 , std::size_t n=  0,
                       std::size_t i = 0, std::size_t j = 0,
                       scalar_t alpha = 1., scalar_t beta = 0.) {
      // if the submatrix is 0x0 then we use the full S matrix
      m = m > 0 ? m : S.get_n_rows();
      n = n > 0 ? n : S.get_n_cols();
      std::size_t cols = M.cols();
      const auto col_ptr_A = S.get_Ac().get_col_ptr();
      const auto row_ind_A = S.get_Ac().get_row_inds();
      const auto col_ptr_B = S.get_Bc().get_col_ptr();
      const auto row_ind_B = S.get_Bc().get_row_inds();
      if (beta == scalar_t(0.))
        A.zero();
      else if (beta != scalar_t(1.))
        A.scale(beta);
      if (alpha == scalar_t(1.)) {
#pragma omp parallel for
        for (std::size_t k=0; k<cols; k++) {
          // iterate through the columns of A, B
          for (size_t c=j; c<j+n; c++) {
            std::size_t startA = col_ptr_A[c],
              endA = col_ptr_A[c+1];
            scalar_t Akc = 0;
            for (std::size_t l=startA; l<endA; l++) {
              std::size_t r = row_ind_A[l] - i;
              if (r >= 0 && r < m)
                Akc += blas::my_conj(M(r,k));
            }
            std::size_t startB = col_ptr_B[c],
              endB = col_ptr_B[c+1];
            for (std::size_t l=startB; l<endB; l++) {
              std::size_t r = row_ind_B[l] - i;
              if (r >= 0 && r < m)
                Akc -= blas::my_conj(M(r,k));
            }
            A(k,c-j) += Akc;
          }
        }
      } else if (alpha == scalar_t(-1.)) {
#pragma omp parallel for
        for (std::size_t k=0; k<cols; k++) {
          // iterate through the columns of A, B
          for(size_t c=j; c<j+n; c++) {
            std::size_t startA = col_ptr_A[c],
              endA = col_ptr_A[c+1];
            scalar_t Akc = 0;
            for(std::size_t l=startA; l<endA; l++) {
              std::size_t r = row_ind_A[l] - i;
              if (r >= 0 && r < m)
                Akc += blas::my_conj(M(r,k));
            }
            std::size_t startB = col_ptr_B[c],
              endB = col_ptr_B[c+1];
            for(std::size_t l=startB; l<endB; l++) {
              std::size_t r = row_ind_B[l] - i;
              if (r >= 0 && r < m)
                Akc -= blas::my_conj(M(r,k));
            }
            A(k,c-j) -= Akc;
          }
        }
      } else {
#pragma omp parallel for
        for(std::size_t k=0; k<cols; k++) {
          //iterate through the columns of A, B
          for(size_t c=j; c<j+n; c++) {
            std::size_t startA = col_ptr_A[c],
              endA = col_ptr_A[c+1];
            scalar_t Akc = 0;
            for(std::size_t l=startA; l<endA; l++) {
              std::size_t r = row_ind_A[l] - i;
              if (r >= 0 && r < m)
                Akc += blas::my_conj(M(r,k));
            }
            std::size_t startB = col_ptr_B[c],
              endB = col_ptr_B[c+1];
            for(std::size_t l=startB; l<endB; l++) {
              std::size_t r = row_ind_B[l] - i;
              if (r >= 0 && r < m)
                Akc -= blas::my_conj(M(r,k));
            }
            A(k,c-j) += alpha * Akc;
          }
        }
      }
    }

  } // namespace HSS
} // namespace strumpack

#endif // HSS_MATRIX_SKETCH_HPP
