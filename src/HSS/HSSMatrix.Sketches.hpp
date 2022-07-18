
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

          BinaryCRSMarix(std::size_t n_cols) {
              n_cols_ = n_cols;
          }

          BinaryCRSMarix(std::vector<std::size_t> col_ind,
              std::vector<std::size_t>row_ptr, std::size_t n_cols) {
              n_cols_ = n_cols;
              //TODO these probably get overridden
              std::size_t nnz_ = col_ind.size();
              std::size_t n_cols_ = row_ptr.size() - 1;
              std::vector< std::size_t> col_ind_(col_ind);
              std::vector< std::size_t> row_ptr_(row_ptr);

          }


          void print() {
              std::cout << "row ptr: ";
              for (std::size_t i = 0; i <= size(); i++)
                  std::cout << row_ptr_[i] << " ";
              std::cout << std::endl << "col ind: ";
              for (std::size_t i = 0; i < nnz(); i++)
                  std::cout << col_ind_[i] << " ";
              std::cout << std::endl;
              std::cout << "val: " << one_ << std::endl;
              std::cout << "n_cols " << n_cols_ << std::endl;
          }

          void print_as_dense() {
              for (std::size_t i = 0; i < row_ptr_.size() - 1; i++) {
                  std::size_t start = row_ptr_[i] - 1,
                   end = row_ptr_[i + 1] - 1;


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
              row_ptr_.push_back(nnz_ + 1);
              //update n_rows
              n_rows_ += 1;
          }



               /*
                * apends the cols of a second B-CRS
                * matrix to the end of this matrix:
                */
              void append_cols(BinaryCRSMarix<scalar_t> T) {
                  if (T.size() != n_rows_) {
                      std::cout << "# Cannot append a matrix with"
                          << " the wrong number of rows" << std::endl;
                      return;
                  }

                  std::vector<std::size_t> new_row_ptr_ = { std::size_t(1) };
                  std::vector<std::size_t> new_col_inds;
                  const std::vector<std::size_t>* rows_T = T.get_row_ptr();
                  const std::vector<std::size_t>* col_T = T.get_col_inds();
                  //update col indices
                  for(std::size_t i = 0; i < row_ptr_.size() - 1; i++) {

                      for (std::size_t j = row_ptr_[i] - 1;
                          j < row_ptr_[i+1] - 1; j++) {
                          new_col_inds.push_back(col_ind_[j]);
                      }
                      for (std::size_t j = (*rows_T)[i] - 1; j <
                      (*rows_T)[i + 1] - 1; j++) {
                          new_col_inds.push_back((*col_T)[j]+n_cols());
                      }
                      new_row_ptr_.push_back(row_ptr_[i + 1] +
                          (*rows_T)[i + 1] - 1);
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

          std::size_t size() {
              return n_rows_;
          }

          std::size_t n_cols() {
              return n_cols_;
          }

          const  std::vector<std::size_t>* get_row_ptr() const{
              return &row_ptr_;
          }

          const  std::vector<std::size_t>* get_col_inds() const{
              return &col_ind_;
          }

      private:
          std::size_t nnz_ = std::size_t(0);
          std::size_t n_cols_;
          std::size_t n_rows_ = std::size_t(0);
          scalar_t one_ = scalar_t(1.);
          std::vector<std::size_t> col_ind_ = {};
          std::vector<std::size_t> row_ptr_ = { std::size_t(1) };
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


       void createSJLTCRS(BinaryCRSMarix<scalar_t> &A,
           BinaryCRSMarix<scalar_t> &B, std::size_t nnz,
           std::size_t n_rows, std::size_t n_cols) {

       if (nnz > n_cols) {
           std::cout << "# POSSIBLE ERROR: nnz bigger than n_cols" << std::endl;
           std::cout << "# setting nnz to n_cols" << std::endl;
           nnz = n_cols;
       }



       //set the nnz value for each of the matrices:
       //A.set_nnz_value(scalar_t(1./std::sqrt(double(nnz))));
       //B.set_nnz_value(scalar_t(-1./std::sqrt(double(nnz))));
       std::vector<std::size_t> col_inds;

       for (std::size_t j = 0; j < n_cols; j++) {
           col_inds.push_back(j);
       }

       std::vector<int> nums = {1,-1};

       for (std::size_t i = 0; i < n_rows; i++) {

           std::vector<std::size_t> a_inds;
           std::vector<std::size_t> b_inds;

           //sample nnz column indices
           std::shuffle(col_inds.begin(), col_inds.end(), e_);

           for (std::size_t j = 0; j < nnz; j++) {

               //decide whether each is +- 1
               std::shuffle(nums.begin(), nums.end(), e_);
               if (nums[0] == 1) {
               //belongs to A
                   a_inds.push_back(col_inds[j]);
               }
               else {
               //belongs to B
                   b_inds.push_back(col_inds[j]);
               }

           }
           //put in A and B row into A and B
           A.add_row(a_inds);
           B.add_row(b_inds);
       }

       }


       void SJLTDenseSketch
       (DenseMatrix<scalar_t>& B, std::size_t nnz) {


           if (nnz > B.cols()) {
               std::cout << "# error nnz too large \n";
               std::cout <<"# n_cols = " << B.cols() << std::endl;
               std::cout <<"# nnz = " << nnz << std::endl;
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
       SJLT_Matrix(SJLTGenerator<scalar_t, integer_t> g, std::size_t nnz, std::size_t n_rows, std::size_t n_cols):
       g_(g), nnz_(nnz), n_rows_(n_rows), n_cols_(n_cols),
       A_(BinaryCRSMarix<scalar_t>(n_cols)),
       B_(BinaryCRSMarix<scalar_t>(n_cols))
       {
           g_.createSJLTCRS(A_, B_, nnz_, n_rows_, n_cols_);

       }

       void add_columns(std::size_t new_cols, std::size_t nnz) {
           if(nnz > new_cols){
               std::cout <<"# nnz bigger than n_cols cannot proceed \n";
              return;
           }
           BinaryCRSMarix<scalar_t> A_temp(new_cols);
           BinaryCRSMarix<scalar_t> B_temp(new_cols);
           g_.createSJLTCRS(A_temp, B_temp, nnz, get_n_rows(), new_cols);
            A_.append_cols(A_temp);
            B_.append_cols(B_temp);
            n_cols_ += new_cols;
            nnz_ += nnz*n_rows_;
       }

       void append_sjlt_matrix(SJLT_Matrix<scalar_t,integer_t> temp){
           if(temp.get_n_rows() != n_rows_){
               std::cout << "# wrong shape to append \n";
           }
           nnz_ +=temp.get_nnz();
           n_cols_ += temp.get_n_cols();
           A_.append_cols(temp.get_A());
           B_.append_cols(temp.get_B());
       }

       void print_SJLT_As_Dense() {
           const std::vector<std::size_t>* rows_A = A_.get_row_ptr();
           const std::vector<std::size_t>* col_A = A_.get_col_inds();
           const std::vector<std::size_t>* rows_B = B_.get_row_ptr();
           const std::vector<std::size_t>* col_B = B_.get_col_inds();

           for (std::size_t i = 0; i < n_rows_ ; i++) {

               std::size_t startA = (*rows_A)[i] - 1,
                endA = (*rows_A)[i + 1] - 1;
               std::size_t startB = (*rows_B)[i] - 1,
               endB = (*rows_B)[i + 1] - 1;

               for (std::size_t j = 0; j < n_cols_; j++) {

                   if (std::find((*col_A).begin() + startA, (*col_A).begin() +
                   endA, j) != (*col_A).begin() + endA) {
                       std::cout << "1 ";
                   }
                   else if (std::find((*col_B).begin() +
                   startB, (*col_B).begin() + endB, j) != (*col_B).begin() +
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

       BinaryCRSMarix<scalar_t>& get_A(){
           return A_;
       }

       BinaryCRSMarix<scalar_t>& get_B(){
           return B_;
       }

       std::size_t get_n_rows() const{
           return n_rows_;
       }

       std::size_t get_n_cols() const{
           return n_cols_;
       }
       std::size_t get_nnz() const{
           return nnz_;
       }
       SJLTGenerator<scalar_t, integer_t>  get_g(){
           return g_;
       }
       //convert SJLT class to densematrix
       DenseMatrix<scalar_t> SJLT_to_dense(){
            DenseMatrix<scalar_t> S(n_rows_, n_cols_);

            const std::vector<std::size_t>* rows_A = A_.get_row_ptr();
            const std::vector<std::size_t>* col_A = A_.get_col_inds();
            const std::vector<std::size_t>* rows_B = B_.get_row_ptr();
            const std::vector<std::size_t>* col_B = B_.get_col_inds();

            for (std::size_t i = 0; i < n_rows_ ; i++) {

                std::size_t startA = (*rows_A)[i] - 1,
                 endA = (*rows_A)[i + 1] - 1;
                std::size_t startB = (*rows_B)[i] - 1,
                endB = (*rows_B)[i + 1] - 1;

                for (std::size_t j = 0; j < n_cols_; j++) {

                    if (std::find((*col_A).begin() + startA, (*col_A).begin() +
                    endA, j) != (*col_A).begin() + endA) {
                        S(i,j) = 1;
                    }
                    else if (std::find((*col_B).begin() +
                    startB, (*col_B).begin() + endB, j) != (*col_B).begin() +
                     endB) {
                        S(i,j) = -1;
                    }
                    else {
                        S(i,j) = 0;
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
       BinaryCRSMarix< scalar_t> B_;
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

 //multiplication
template<typename scalar_t, typename integer_t> DenseMatrix<scalar_t>
Matrix_times_SJLT(const DenseMatrix<scalar_t>& M ,
    SJLT_Matrix<scalar_t, integer_t>& S)
         {
             DenseMatrix<scalar_t> D(S.get_n_rows(), S.get_n_cols());

             const std::vector<std::size_t>* rows_A = S.get_A().get_row_ptr();
             const std::vector<std::size_t>* col_A = S.get_A().get_col_inds();
             const std::vector<std::size_t>* rows_B = S.get_B().get_row_ptr();
             const std::vector<std::size_t>* col_B = S.get_B().get_col_inds();

             for(size_t i = 0; i <(*rows_A).size() - 1 ;i++){

                    size_t start_A = (*rows_A)[i] - 1, end_A = (*rows_A)[i+1] -1;

                  for(std::size_t j =start_A;j < end_A; j++){

                      pm_column<scalar_t>(M,i, D, (*col_A)[j], true);
                  }

                  //subtract cols
                 std::size_t startB = (*rows_B)[i] - 1,endB = (*rows_B)[i + 1] - 1;

                 for(std::size_t j =startB; j < endB; j++){
                     pm_column<scalar_t>(M,i, D, (*col_B)[j], false);
                 }

             }

         return D;
        }




    }
    }
#endif
