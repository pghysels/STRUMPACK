
#ifndef HSS_MATRIX_SKETCHES
#define HSS_MATRIX_SKETCHES


#include "misc/RandomWrapper.hpp"
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>

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

               std::vector<scalar_t> nums = {scalar_t(1./std::sqrt(double(nnz))),
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



  }
}
#endif
