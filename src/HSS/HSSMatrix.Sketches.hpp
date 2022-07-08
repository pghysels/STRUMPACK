
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

      class SJLT{
      public:

          SJLT(){
              seed_ =
              std::chrono::system_clock::now().time_since_epoch().count();
              e_.seed(seed_);
          }

          SJLT(unsigned long seed){
              seed_ = seed;
               e_.seed(seed_);

          }

          void set_seed(unsigned long seed){
              seed_ = seed;
               e_.seed(seed_);
          }



          template<typename scalar_t>  void SJLTSketch
          (DenseMatrix<scalar_t>& B, std::size_t nnz){


               if(nnz >= B.cols()){
                   std::cout << "nnz too large \n";
                   return; //either make error or make nnz - B.cols()
               }
               //set initial B to zero:
               B.zero();
               std::vector<int> col_inds;

               for (unsigned int j = 0; j < B.cols(); j++){
                   col_inds.push_back(j);
               }

               std::vector<scalar_t> nums = {
                   scalar_t(1./std::sqrt(double(nnz))),
                   scalar_t(-1./std::sqrt(double(nnz)))};

                for (std::size_t i=0; i<B.rows(); i++){
                    //sample nnz column indices breaks in second loop here
                    //take the first nnz elements nonzero, else 0
                    std::shuffle(col_inds.begin(), col_inds.end(),e_);
                    for (std::size_t j=0; j<nnz; j++){
                        //decide whether each is +- 1
                        std::shuffle(nums.begin(), nums.end(),e_);
                        B(i,col_inds[j]) = nums[0];
                        }
                }
         }

         /*
         * function which will take a matrix A, an SJLT matrix S
         * and write A*S to AS
         */
         template<typename scalar_t>  void multiplySJLT
         (DenseMatrix<scalar_t>& A, DenseMatrix<scalar_t>& S,
         DenseMatrix<scalar_t>& AS){
             //TODO
         }

     private:
          unsigned long seed_;
          std::default_random_engine e_;
      };



      template<typename scalar_t, typename integer_t> class BinaryCRSMarix {
      public:

          BinaryCRSMarix(integer_t n_cols) {
              n_cols_ = n_cols;
          }

          BinaryCRSMarix(std::vector<integer_t> col_ind,
              std::vector<integer_t>row_ptr, integer_t n_cols) {
              n_cols_ = n_cols;
              //TODO
              /*
              integer_t nnz_ = row_ind.size();
              integer_t n_cols_ = col_ptr.size() - 1;
              std::vector<integer_t> row_ind_(row_ind);
              std::vector<integer_t> col_ptr_(col_ptr);
              */
          }


          void print() {
              std::cout << "row ptr: ";
              for (integer_t i = 0; i <= size(); i++)
                  std::cout << row_ptr_[i] << " ";
              std::cout << std::endl << "col ind: ";
              for (integer_t i = 0; i < nnz(); i++)
                  std::cout << col_ind_[i] << " ";
              std::cout << std::endl;
              std::cout << "val: " << one_ << std::endl;
              std::cout <<"n_cols " << n_cols_ << std::endl;
          }

          void print_as_dense() {
              for (integer_t i = 0; i < row_ptr_.size() - 1; i++) {
                  integer_t start = row_ptr_[i] - 1, end = row_ptr_[i + 1] -1;


                  for (integer_t j = 0; j < n_cols_; j++) {

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
              std::vector<integer_t> new_col_ind_) {
              integer_t added_nnz_ = new_col_ind_.size();
              //append col_inds
              col_ind_.insert(std::end(col_ind_),
                  std::begin(new_col_ind_), std::end(new_col_ind_));
              //update nnz_
              nnz_ += added_nnz_;
              //update row_ptr
              row_ptr_.push_back(nnz_ + 1);
              //update n_rows
              n_rows_ += 1;
          }

          void set_nnz_value(scalar_t one) {
              one_ = one;
          }

          scalar_t nnz_value() {
              return one_;
          }

          integer_t nnz() {
              return nnz_;
          }

          integer_t size() {
              return n_rows_;
          }
      private:
          integer_t nnz_ = integer_t(0);
          integer_t n_cols_;
          integer_t n_rows_ = integer_t(0);
          scalar_t one_ = scalar_t(1.);
          std::vector<integer_t> col_ind_ = {};
          std::vector<integer_t> row_ptr_ = { integer_t(1) };
      };


template<typename scalar_t,typename integer_t> class SJLT_Matrix{
public:

// Print A
// print B
// print sjlt
// apply SJLT_matrix to A*SJLT
private:
    // 1 will be cast as scalar_t 1
    //integer_t for size of row/column indices
    BinaryCRSMarix< scalar_t, integer_t> A_row_wise;
    BinaryCRSMarix< scalar_t, integer_t> B_row_wise;
    scalar_t nnz;
};


  }
}
#endif
