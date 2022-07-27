
#ifndef HSS_MATRIX_SKETCHES
#define HSS_MATRIX_SKETCHES


#include "misc/RandomWrapper.hpp"
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <stdexcept>

namespace strumpack {
  namespace HSS {



      template<typename scalar_t> class BinaryCRSMarix {


      public:

          BinaryCRSMarix(std::size_t n_rows, std::size_t n_cols) :
              nnz_(std::size_t(0)), n_cols_(n_cols), n_rows_(n_rows),
              one_(scalar_t(1.)), col_ind_({}), row_ptr_({ std::size_t(0) }) {}

          BinaryCRSMarix(std::vector<std::size_t> col_ind,
              std::vector<std::size_t>row_ptr, std::size_t n_cols) :
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
              for (std::size_t i = 0; i < row_ptr_.size() - 1; i++) {
                  std::size_t start = row_ptr_[i],
                      end = row_ptr_[i + 1];


                  for (std::size_t j = 0; j < n_cols_; j++) {

                      if (std::find(col_ind_.begin() + start, col_ind_.begin()
                          + end, j) != col_ind_.begin() + end) {
                          std::cout << one_ << " ";
                      }
                      else {
                          std::cout << "0 ";
                      }

                  }
                  std::cout << std::endl;
              }
          }

          void add_row(
              std::vector<std::size_t> new_col_ind_) {
              std::size_t added_nnz_ = new_col_ind_.size();
              //append col_inds
              col_ind_.insert(std::end(col_ind_),
                  std::begin(new_col_ind_), std::end(new_col_ind_));
              //update nnz_
              nnz_ += added_nnz_;
              //update row_ptr
              row_ptr_.push_back(nnz_);
              //update n_rows
              n_rows_ += 1;
          }



          /*
           * apends the cols of a second B-CRS
           * matrix to the end of this matrix:
           */
          void append_cols(BinaryCRSMarix<scalar_t>& T) {
              if (T.n_rows() != n_rows_) {
                  std::cout << "# Cannot append a matrix with"
                      << " the wrong number of rows" << std::endl;
                  std::cout << "# original rows: " << n_rows_
                      << "  new rows: " << T.n_rows() << std::endl;
                  return;
              }

              const auto rows_T = T.get_row_ptr();
              const auto col_T = T.get_col_inds();

              std::vector<std::size_t> new_row_ptr_;
              new_row_ptr_.reserve(rows_T.size());
              new_row_ptr_.push_back(std::size_t(0));

              std::vector<std::size_t> new_col_inds;
              new_col_inds.reserve(col_T.size() + col_ind_.size());

              //update col indices
              for (std::size_t i = 0; i < row_ptr_.size() - 1; i++) {

                  for (std::size_t j = row_ptr_[i];
                      j < row_ptr_[i + 1]; j++) {
                      new_col_inds.push_back(col_ind_[j]);
                  }
                  for (std::size_t j = rows_T[i]; j <
                      rows_T[i + 1]; j++) {
                      new_col_inds.push_back(col_T[j] + n_cols());
                  }
                  new_row_ptr_.push_back(row_ptr_[i + 1] +
                      rows_T[i + 1]);
              }

              nnz_ += T.nnz();
              n_cols_ += T.n_cols();
              col_ind_ = new_col_inds;
              row_ptr_ = new_row_ptr_;

          }

          void set_nnz_value(scalar_t one) {
              one_ = one;
          }

          scalar_t nnz_value() {
              return one_;
          }

          std::size_t nnz() {
              return nnz_;
          }

          std::size_t n_rows() {
              return n_rows_;
          }

          std::size_t n_cols() {
              return n_cols_;
          }

          const  std::vector<std::size_t>& get_row_ptr() const {
              return row_ptr_;
          }

          const  std::vector<std::size_t>& get_col_inds() const {
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
          std::size_t nnz_;
          std::size_t n_cols_;
          std::size_t n_rows_;
          scalar_t one_;
          std::vector<std::size_t> col_ind_;
          std::vector<std::size_t> row_ptr_;
      };


      template<typename scalar_t> class BinaryCCSMarix {
      public:

          BinaryCCSMarix(std::size_t n_rows, std::size_t n_cols) :
              nnz_(std::size_t(0)), n_cols_(n_cols), n_rows_(n_rows),
              one_(scalar_t(1.)), row_ind_({}), col_ptr_({ std::size_t(0) }) {}

          BinaryCCSMarix(std::vector<std::size_t> row_ind,
              std::vector<std::size_t>col_ptr, std::size_t n_rows) :
              nnz_(row_ind.size()), n_cols_(col_ptr.size() - 1),
              n_rows_(n_rows), one_(scalar_t(1.)),
              row_ind_(row_ind), col_ptr_(col_ptr) {}


          void print() {
              std::cout << "col ptr: ";
              for (std::size_t i = 0; i < col_ptr_.size(); i++)
                  std::cout << col_ptr_[i] << " ";
              std::cout << std::endl << "row ind: ";
              for (std::size_t i = 0; i < row_ind_.size(); i++)
                  std::cout << row_ind_[i] << " ";
              std::cout << std::endl << "val: ";
              std::cout << one_ << " ";
              std::cout << std::endl;
          }

          void print_as_dense() {
              std::vector<std::string> vec(n_rows(), "");

              for (std::size_t i = 0; i < col_ptr_.size() - 1; i++) {
                  std::size_t start = col_ptr_[i], end = col_ptr_[i + 1];

                  for (std::size_t j = 0; j < n_rows(); j++) {

                      if (std::find(row_ind_.begin() + start,
                          row_ind_.begin() + end, j) != row_ind_.begin() + end){
                          vec[j] += std::to_string(one_) + " ";
                      }
                      else {
                          vec[j] += "0 ";
                      }

                  }

              }

              for (std::size_t i = 0; i < n_rows(); i++) {
                  std::cout << vec[i] << std::endl;
              }

          }

          void append_cols(BinaryCCSMarix<scalar_t>& T) {

              if (T.n_rows() != n_rows_) {
                  std::cout << "# Cannot append a matrix with"
                      << " the wrong number of rows" << std::endl;
                  return;
              }

              const auto new_cols = T.get_col_ptr();
              const auto new_row_inds = T.get_row_inds();

              row_ind_.reserve(row_ind_.size() + new_row_inds.size());
              row_ind_.insert(row_ind_.end(),
                  new_row_inds.begin(), new_row_inds.end());

              //add new columns:
              col_ptr_.reserve(col_ptr_.size() + new_cols.size());

              for (auto i : new_cols) {
                  if (i != std::size_t(0))
                      col_ptr_.push_back(i + nnz_);
              }


              //one and n_rows_ does not change
              nnz_ += T.nnz();
              n_cols_ += T.n_cols();
          }

          void add_col(
              std::vector<std::size_t> new_row_ind_) {
              std::size_t added_nnz_ = new_row_ind_.size();
              //append col_inds
              row_ind_.insert(std::end(row_ind_),
                  std::begin(new_row_ind_), std::end(new_row_ind_));
              //update nnz_
              nnz_ += added_nnz_;
              //update row_ptr
              col_ptr_.push_back(nnz_);
              //update n_rows
              n_cols_ += 1;
          }


          void set_nnz_value(scalar_t one) {
              one_ = one;
          }

          scalar_t nnz_value() {
              return one_;
          }

          std::size_t nnz() {
              return nnz_;
          }

          std::size_t n_cols() {
              return n_cols_;
          }

          std::size_t n_rows() {
              return n_rows_;
          }

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
              //update nnz + n_rows
          }

      private:
          std::size_t nnz_;
          std::size_t n_cols_;
          std::size_t n_rows_;
          scalar_t one_;
          std::vector<std::size_t> row_ind_;
          std::vector<std::size_t> col_ptr_;
      };




      template<typename scalar_t, typename integer_t>  class SJLTGenerator {
      public:

          SJLTGenerator() {
              seed_ =
                  std::chrono::system_clock::now().
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
              BinaryCRSMarix<scalar_t>& B, BinaryCCSMarix<scalar_t>& Ac,
              BinaryCCSMarix<scalar_t>& Bc, std::size_t nnz,
              std::size_t n_rows, std::size_t n_cols) {

              if (nnz > n_cols) {
                  std::cout << "# POSSIBLE ERROR: nnz bigger than n_cols"
                            << std::endl;
                  std::cout << "# setting nnz to n_cols" << std::endl;
                  nnz = n_cols;
              }


              //set the nnz value for each of the matrices:
              A.set_nnz_value(scalar_t(1));
              Ac.set_nnz_value(scalar_t(1));
              B.set_nnz_value(scalar_t(-1));
              Bc.set_nnz_value(scalar_t(-1));

              //We'll be generating 8 pointers for the 4 matrices:

              //rowise:

              std::vector<std::size_t> A_row_ptr(1 + n_rows,0);
              std::vector<std::size_t> A_col_inds;


              A_col_inds.reserve(n_rows * nnz);

              std::vector<std::size_t> B_row_ptr(1 + n_rows, 0);
              std::vector<std::size_t> B_col_inds;

              B_col_inds.reserve(n_rows * nnz);

              //columnwise:

              std::vector<std::size_t> Ac_col_ptr(1 + n_cols, 0);
              std::vector<std::vector<std::size_t>>
              Ac_row_inds(n_cols, std::vector<std::size_t>());

              for (auto v : Ac_row_inds)
                  v.reserve(std::size_t(nnz*n_rows/n_cols + 1));

              std::vector<std::size_t> Bc_col_ptr(1 + n_cols, 0);
              std::vector<std::vector<std::size_t>>
              Bc_row_inds(n_cols, std::vector<std::size_t>());

              for (auto v : Bc_row_inds)
                  v.reserve(std::size_t(nnz * n_rows / n_cols + 1));


              //SJLT generation algorithm:

              std::vector<std::size_t> col_inds;
              col_inds.reserve(n_cols);

              for (std::size_t j = 0; j < n_cols; j++) {
                  col_inds.push_back(j);
              }

              std::vector<int> nums = { 1,-1 };
              std::size_t a_nnz = 0, b_nnz = 0;

              for (std::size_t i = 0; i < n_rows; i++) {

                  //sample nnz column indices
                  std::shuffle(col_inds.begin(), col_inds.end(), e_);

                  a_nnz = 0, b_nnz = 0;

                  for (std::size_t j = 0; j < nnz; j++) {

                      //decide whether each is +- 1
                      std::shuffle(nums.begin(), nums.end(), e_);
                      if (nums[0] == 1) {
                          //belongs to A
                          A_col_inds.push_back(col_inds[j]);
                          a_nnz++;

                          //CCS processing:
                          Ac_col_ptr[col_inds[j]+1]++;
                          Ac_row_inds[col_inds[j]].push_back(i);
                      }
                      else {
                          //belongs to B
                          B_col_inds.push_back(col_inds[j]);
                          b_nnz++;

                          //CCS processing:
                          Bc_col_ptr[col_inds[j]+1]++;
                          Bc_row_inds[col_inds[j]].push_back(i);
                      }

                  }
                  //put in A and B row into A and B
                  A_row_ptr[i + 1] = A_row_ptr[i] + a_nnz;
                  B_row_ptr[i + 1] = B_row_ptr[i] + b_nnz;
              }

              A.set_ptrs(A_col_inds,A_row_ptr);
              B.set_ptrs(B_col_inds, B_row_ptr);

              //Columnwise processing:

              //update col_ptr by summing previous indices:
              for (std::size_t i = 1; i < Ac_col_ptr.size(); i++)
                  Ac_col_ptr[i] += Ac_col_ptr[i-1];

              for (std::size_t i = 1; i < Bc_col_ptr.size(); i++)
                  Bc_col_ptr[i] += Bc_col_ptr[i - 1];



              //update row_inds by unravelling vectors:
              std::vector<std::size_t> Ac_final_inds;
              Ac_final_inds.reserve(Ac_col_ptr[Ac_col_ptr.size() - 1]);

              for (auto&& v : Ac_row_inds)
                  Ac_final_inds.insert(Ac_final_inds.end(), v.begin(), v.end());


              std::vector<std::size_t> Bc_final_inds;
              Bc_final_inds.reserve(Bc_col_ptr[Ac_col_ptr.size() - 1]);

              for (auto&& v : Bc_row_inds)
                  Bc_final_inds.insert(Bc_final_inds.end(), v.begin(), v.end());

              Ac.set_ptrs(Ac_final_inds,Ac_col_ptr);
              Bc.set_ptrs(Bc_final_inds, Bc_col_ptr);
          }



          void SJLTDenseSketch
          (DenseMatrix<scalar_t>& B, std::size_t nnz) {


              if (nnz > B.cols()) {
                  std::cout << "# error nnz too large \n";
                  std::cout << "# n_cols = " << B.cols() << std::endl;
                  std::cout << "# nnz = " << nnz << std::endl;
                  return; //either make error or make nnz - B.cols()
              }
              //set initial B to zero:
              B.zero();
              std::vector<int> col_inds;

              for (unsigned int j = 0; j < B.cols(); j++) {
                  col_inds.push_back(j);
              }

              std::vector<scalar_t> nums = {
                 scalar_t(1.),
                  scalar_t(-1.) };

              for (std::size_t i = 0; i < B.rows(); i++) {
                  //sample nnz column indices breaks in second loop here
                  //take the first nnz elements nonzero, else 0
                  std::shuffle(col_inds.begin(), col_inds.end(), e_);
                  for (std::size_t j = 0; j < nnz; j++) {
                      //decide whether each is +- 1
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
template<typename scalar_t, typename integer_t> class SJLT_Matrix {
public:
    SJLT_Matrix(SJLTGenerator<scalar_t, integer_t> g, std::size_t nnz,
        std::size_t n_rows, std::size_t n_cols) :
        g_(g), nnz_(nnz), n_rows_(n_rows), n_cols_(n_cols),
        A_(BinaryCRSMarix<scalar_t>(0, n_cols)),
        B_(BinaryCRSMarix<scalar_t>(0, n_cols)),
        Ac_(BinaryCCSMarix<scalar_t>(n_rows, 0)),
        Bc_(BinaryCCSMarix<scalar_t>(n_rows, 0))
    {
        g_.createSJLTCRS(A_, B_, Ac_, Bc_, nnz_, n_rows_, n_cols_);

    }

    void add_columns(std::size_t new_cols, std::size_t nnz) {
        if (nnz > new_cols) {
            std::cout << "# nnz bigger than n_cols cannot proceed \n";
            return;
        }
        BinaryCRSMarix<scalar_t> A_temp(0, new_cols);
        BinaryCRSMarix<scalar_t> B_temp(0, new_cols);
        /*Fix this*/
        BinaryCCSMarix<scalar_t> Ac_temp(n_rows_, 0);
        BinaryCCSMarix<scalar_t> Bc_temp(n_rows_, 0);

        g_.createSJLTCRS(A_temp, B_temp, Ac_temp, Bc_temp,
                         nnz, n_rows_, new_cols);
        A_.append_cols(A_temp);
        B_.append_cols(B_temp);
        Ac_.append_cols(Ac_temp);
        Bc_.append_cols(Bc_temp);
        n_cols_ += new_cols;
        nnz_ += nnz * n_rows_;
    }

    void append_sjlt_matrix(SJLT_Matrix<scalar_t, integer_t> temp) {
        if (temp.get_n_rows() != n_rows_) {
            std::cout << "# wrong shape to append \n";
        }
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

        for (std::size_t i = 0; i < n_rows_; i++) {

            std::size_t startA = rows_A[i],
                endA = rows_A[i + 1];
            std::size_t startB = rows_B[i],
                endB = rows_B[i + 1];

            for (std::size_t j = 0; j < n_cols_; j++) {

                if (std::find(col_A.begin() + startA, col_A.begin() +
                    endA, j) != col_A.begin() + endA) {
                    std::cout << "1 ";
                }
                else if (std::find(col_B.begin() +
                    startB, col_B.begin() + endB, j) != col_B.begin() +
                    endB) {
                    std::cout << "-1 ";
                }
                else {
                    std::cout << "0 ";
                }

            }
            std::cout << std::endl;
        }

    }

    BinaryCRSMarix<scalar_t>& get_A() {
        return A_;
    }

    BinaryCRSMarix<scalar_t>& get_B() {
        return B_;
    }

    BinaryCCSMarix<scalar_t>& get_Ac() {
        return Ac_;
    }

    BinaryCCSMarix<scalar_t>& get_Bc() {
        return Bc_;
    }

    std::size_t get_n_rows() const {
        return n_rows_;
    }

    std::size_t get_n_cols() const {
        return n_cols_;
    }
    std::size_t get_nnz() const {
        return nnz_;
    }
    SJLTGenerator<scalar_t, integer_t> & get_g() {
        return g_;
    }


    //convert SJLT class to densematrix
    DenseMatrix<scalar_t> SJLT_to_dense() {
        DenseMatrix<scalar_t> S(n_rows_, n_cols_);

        const auto rows_A = A_.get_row_ptr();
        const auto col_A = A_.get_col_inds();
        const auto rows_B = B_.get_row_ptr();
        const auto col_B = B_.get_col_inds();

        for (std::size_t i = 0; i < n_rows_; i++) {

            std::size_t startA = rows_A[i],
                endA = rows_A[i + 1];
            std::size_t startB = rows_B[i],
                endB = rows_B[i + 1];

            for (std::size_t j = 0; j < n_cols_; j++) {

                if (std::find(col_A.begin() + startA, col_A.begin() +
                    endA, j) != col_A.begin() + endA) {
                    S(i, j) = 1;
                }
                else if (std::find(col_B.begin() +
                    startB, col_B.begin() + endB, j) != col_B.begin() +
                    endB) {
                    S(i, j) = -1;
                }
                else {
                    S(i, j) = 0;
                }

            }
        }
        return S;
    }


private:
    SJLTGenerator<scalar_t, integer_t> g_;
    std::size_t nnz_;
    std::size_t n_rows_;
    std::size_t n_cols_;
    BinaryCRSMarix< scalar_t> A_;
    BinaryCRSMarix<scalar_t> B_;
    BinaryCCSMarix<scalar_t> Ac_;
    BinaryCCSMarix<scalar_t> Bc_;
};


// helper functions to do an efficient multiplication of M*S

/*
* add or subtract column based on 'plus' of col_get of M to column col_put of D
*/
 template<typename scalar_t> void pm_column(const DenseMatrix<scalar_t>& M,
      std::size_t col_get, DenseMatrix<scalar_t>& D, std::size_t col_put,
      bool plus){

     if( col_get >= M.cols() || col_put >= D.cols()){
         std::cout<< "# Error index out of bounds\n";
     }

     for(std::size_t r = 0; r < M.rows(); r++){

         D(r, col_put) += plus ?  M(r,col_get): - M(r,col_get);
     }
 }

 //multiplication AS
template<typename scalar_t, typename integer_t> void
Matrix_times_SJLT(const DenseMatrix<scalar_t>& M ,
    SJLT_Matrix<scalar_t, integer_t>& S, DenseMatrix<scalar_t>& A)
         {
             A.zero();
             const auto rows_A = S.get_A().get_row_ptr();
             const auto col_A = S.get_A().get_col_inds();
             const auto rows_B = S.get_B().get_row_ptr();
             const auto col_B = S.get_B().get_col_inds();

             for(size_t i = 0; i <rows_A.size() - 1 ;i++){

                    size_t start_A = rows_A[i], end_A = rows_A[i + 1];

                  for(std::size_t j =start_A;j < end_A; j++){
                      pm_column<scalar_t>(M,i, A, col_A[j], true);
                  }

                  //subtract cols
                 std::size_t startB = rows_B[i],endB = rows_B[i + 1];

                 for(std::size_t j =startB; j < endB; j++){
                     pm_column<scalar_t>(M,i, A, col_B[j], false);
                 }

             }

        }

/*
        //multiplication M^TS, A <- answer
       template<typename scalar_t, typename integer_t> void
       MatrixT_times_SJLT(const DenseMatrix<scalar_t>& M ,
           SJLT_Matrix<scalar_t, integer_t>& S, DenseMatrix<scalar_t>& A)
                {
                    A.zero();
                    const auto rows_A = S.get_A().get_row_ptr();
                    const auto col_A = S.get_A().get_col_inds();
                    const auto rows_B = S.get_B().get_row_ptr();
                    const auto col_B = S.get_B().get_col_inds();

                    for(size_t i = 0; i <rows_A.size() - 1 ;i++){

                           size_t start_A = rows_A[i], end_A = rows_A[i + 1];

                         for(std::size_t j =start_A;j < end_A; j++){
                             pm_column<scalar_t>(M,i, A, col_A[j], true);
                         }

                         //subtract cols
                        std::size_t startB = rows_B[i],endB = rows_B[i + 1];

                        for(std::size_t j =startB; j < endB; j++){
                            pm_column<scalar_t>(M,i, A, col_B[j], false);
                        }

                    }

               }
*/

    }
    }
#endif
