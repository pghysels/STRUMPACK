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
#ifndef DIST_SAMPLES_HPP
#define DIST_SAMPLES_HPP

#include "HSSOptions.hpp"
#include "HSS/HSSMatrix.sketch.hpp"
#include <chrono>


namespace strumpack {
  namespace HSS {

    template<typename scalar_t> class HSSMatrixMPI;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
    template<typename scalar_t> class DistSamples {
      using real_t = typename RealType<scalar_t>::value_type;
      using DistM_t = DistributedMatrix<scalar_t>;
      using DistMW_t = DistributedMatrixWrapper<scalar_t>;
      using DenseM_t = DenseMatrix<scalar_t>;
      using dmult_t = typename std::function
        <void(DistM_t& R, DistM_t& Sr, DistM_t& Sc)>;
      using opts_t = HSSOptions<scalar_t>;

    public:
      DistSamples() {}
      DistSamples(const BLACSGrid* g, const dmult_t& Amult,
                  HSSMatrixMPI<scalar_t>& hss, const opts_t& opts)
        : DistSamples(g, hss, opts) {
        Amult_ = &Amult;
      }
      DistSamples(const DistM_t& A, HSSMatrixMPI<scalar_t>& hss,
                  const opts_t& opts)
        : DistSamples(A.grid(), hss, opts) {
        A_= &A;
        auto begin = std::chrono::steady_clock::now();
        hss.to_block_row(A, sub_A, leaf_A);
        
        auto end = std::chrono::steady_clock::now();
        auto T = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        if(hss_ -> Comm().is_root())
        std::cout << "# total 2dbc to block row redistribution = " << T << " [10e-3s]" << std::endl;
      
        if (leaf_A.rows())
          std::cout << "ERROR: Not supported. "
                    << " Make sure there are more MPI ranks than HSS leafs."
                    << std::endl;
      }
      DistSamples(int d, const BLACSGrid* g, const dmult_t& Amult,
                  HSSMatrixMPI<scalar_t>& hss, const opts_t& opts)
        : DistSamples(g, Amult, hss, opts) {
        add_columns(d, opts);
      }
      DistSamples(int d, const DistM_t& A, HSSMatrixMPI<scalar_t>& hss,
                  const opts_t& opts)
        : DistSamples(A, hss, opts) {
        add_columns(d, opts);
      }

      int cols() const { return d_; }

      const HSSMatrixMPI<scalar_t>& HSS() const { return *hss_; }

      void add_columns(int d, const opts_t& opts) {
        auto d_old = d_;
        auto dd = d - d_old;
        d_ = d;
        DenseM_t subRnew, subSrnew, subScnew;
        

        if (Amult_) {
          int rank = g_ -> Comm().rank();

          if (!rank) {
            if (opts.verbose())
              std::cout << "# sampling with 2DBC matrix" << std::endl;
            if (opts.compression_sketch() == CompressionSketch::SJLT)
              std::cout << "WARNING: SJLT sampling is not supported for 2DBC layout"
                        << std::endl;
          }
          
          if (!d_old) {
            R = DistM_t(g_, n_, d_);
            Sr = DistM_t(g_, n_, d_);
            Sc = DistM_t(g_, n_, d_);
            rgen_->seed(R.prow(), R.pcol());
            R.random(*rgen_);
            STRUMPACK_RANDOM_FLOPS
            (rgen_->flops_per_prng() * R.lrows() * R.lcols());
            

            g_->Comm().barrier();
            
             auto begin = std::chrono::steady_clock::now();
            (*Amult_)(R, Sr, Sc);
            
            g_->Comm().barrier();
            auto end = std::chrono::steady_clock::now();
            auto T = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
            
            if (!rank)
            std::cout << "# Gaussian init multiplies = " << T << " [10e-3s]" << std::endl;



            Sc = Sr;
            
            hss_->to_block_row(R,  sub_Rr, leaf_R);
            hss_->to_block_row(Sr, sub_Sr, leaf_Sr);
            hss_->to_block_row(Sc, sub_Sc, leaf_Sc);
            sub_Rc = sub_Rr;
          } else {
            DistM_t Rnew(g_, n_, dd), Srnew(g_, n_, dd), Scnew(g_, n_, dd);
            Rnew.random(*rgen_);
            STRUMPACK_RANDOM_FLOPS
              (rgen_->flops_per_prng() * Rnew.lrows() * Rnew.lcols());
            


            g_->Comm().barrier();
            
            auto begin = std::chrono::steady_clock::now();
            (*Amult_)(Rnew, Srnew, Scnew);
            
            g_->Comm().barrier();
            auto end = std::chrono::steady_clock::now();
            auto T = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
            
            if (!rank)
            std::cout << "# Gaussian add_cols multiplies = " << T << " [10e-3s]" << std::endl;

                        
            Scnew = Srnew;


            R.hconcat(Rnew);
            Sr.hconcat(Srnew);
            Sc.hconcat(Scnew);
            DistM_t leafRnew, leafSrnew, leafScnew;
            hss_->to_block_row(Rnew,  subRnew, leafRnew);
            hss_->to_block_row(Srnew, subSrnew, leafSrnew);
            hss_->to_block_row(Scnew, subScnew, leafScnew);
            leaf_R.hconcat(leafRnew);
            leaf_Sr.hconcat(leafSrnew);
            leaf_Sc.hconcat(leafScnew);
          }
        } else {

          auto rank = hss_->Comm().rank();
          if (opts.verbose() && hss_->Comm().is_root())
            std::cout << "# sampling with 1DBR matrix" << std::endl;

          auto r0 = hss_->tree_ranges().clo(rank);
          auto r1 = hss_->tree_ranges().chi(rank);
          bool chunk = opts.SJLT_algo() == SJLTAlgo::CHUNK;
          SJLTGenerator<scalar_t,int> g(d_);
          SJLTMatrix<scalar_t,int> S_sjlt(g, 0, n_, 0, chunk);
          if (!d_old) {
            sub_Sr = DenseM_t(sub_A.rows(), d_);
            if (opts.compression_sketch() == CompressionSketch::SJLT) {
              S_sjlt.add_columns(d_, opts.nnz0());
      
              hss_->Comm().barrier();
              auto begin = std::chrono::steady_clock::now();
              matrix_times_SJLT(sub_A, S_sjlt, sub_Sr);

              hss_->Comm().barrier();
              auto end = std::chrono::steady_clock::now();
              auto T = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
              
              if(!rank){
                std::cout << "# SJLT init multiplies = " << T << " [10e-3s]" << std::endl;
 
              }
              
      
      
              sub_Rr = S_sjlt.to_dense_sub_block(r1-r0, d_, r0, 0);
            } else {
              DenseM_t dup_R(n_, d_);
              rgen_->seed(0, 0);
              dup_R.random(*rgen_);

              auto begin = std::chrono::steady_clock::now();
              gemm(Trans::N, Trans::N, scalar_t(1.), sub_A, dup_R, scalar_t(0.), sub_Sr);
              
              auto end = std::chrono::steady_clock::now();
              auto T = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
              
              int minT, maxT;
              MPI_Reduce(&T, &minT, 1, MPI_INT, MPI_MIN, 0, hss_->comm());
              MPI_Reduce(&T, &maxT, 1, MPI_INT, MPI_MAX, 0, hss_->comm());
              if(!rank == 0){
                std::cout << "# Gaussian init multiplies min= " << minT << " [10e-3s]" << std::endl;
                std::cout << "# Gaussian init multiplies max= " << maxT << " [10e-3s]" << std::endl;
 
              }
              
      
              sub_Rr = DenseM_t(r1-r0, d_, dup_R, r0, 0);
            }
            // TODO transpose, assume for now matrix is symmetric
            sub_Sc = sub_Sr;
            sub_Rc = sub_Rr;
          } else {
            subSrnew = DenseM_t(sub_A.rows(), dd);
            if (opts.compression_sketch() == CompressionSketch::SJLT) {
              SJLTGenerator<scalar_t,int> g(d_);
              SJLTMatrix<scalar_t,int> S_sjlt(g, 0, n_, 0, chunk);
              S_sjlt.add_columns(dd, opts.nnz0());
              
              hss_->Comm().barrier();
              auto begin = std::chrono::steady_clock::now();
        
              matrix_times_SJLT(sub_A, S_sjlt, subSrnew);
              
              hss_->Comm().barrier();
              auto end = std::chrono::steady_clock::now();

              auto T = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
              
              if(!rank){
                std::cout << "# SJLT add_cols multiplies = " << T << " [10e-3s]" << std::endl;
              }

              subRnew = S_sjlt.to_dense_sub_block(r1-r0, dd, r0, 0);
            } else {
              DenseM_t dup_R(n_, dd);
              rgen_->seed(0, 0);
              dup_R.random(*rgen_);
      
              auto begin = std::chrono::steady_clock::now();
              gemm(Trans::N, Trans::N, scalar_t(1.), sub_A, dup_R, scalar_t(0.), subSrnew);
            
              auto end = std::chrono::steady_clock::now();
              auto T = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
              
              int minT, maxT;
              MPI_Reduce(&T, &minT, 1, MPI_INT, MPI_MIN, 0, hss_->comm());
              MPI_Reduce(&T, &maxT, 1, MPI_INT, MPI_MAX, 0, hss_->comm());
              if(rank == 0){
                std::cout << "# Gaussian add_cols multiplies min= " << minT << " [10e-3s]" << std::endl;
                std::cout << "# Gaussian add_cols multiplies max= " << maxT << " [10e-3s]" << std::endl;
 
              }
              subRnew = DenseM_t(r1-r0, dd, dup_R, r0, 0);
            }
            // TODO transpose, assume for now matrix is symmetric
            subScnew = subSrnew;
          }
        }
        if (d_old) {
          if (hard_restart) {
            sub_Rr = hconcat(sub_R2,  subRnew);
            sub_Rc = hconcat(sub_R2,  subRnew);
            sub_Sr = hconcat(sub_Sr2, subSrnew);
            sub_Sc = hconcat(sub_Sc2, subScnew);
            sub_R2  = sub_Rr;
            sub_Sr2 = sub_Sr;
            sub_Sc2 = sub_Sc;
          } else {
            sub_Rr.hconcat(subRnew);
            sub_Rc.hconcat(subRnew);
            sub_Sr.hconcat(subSrnew);
            sub_Sc.hconcat(subScnew);
          }
        }
        if (hard_restart) {
          sub_R2 = sub_Rr;
          sub_Sr2 = sub_Sr;
          sub_Sc2 = sub_Sc;
        }
      }

      DistM_t leaf_R, leaf_Sr, leaf_Sc;
      DenseM_t sub_Rr, sub_Rc, sub_Sr, sub_Sc;
      bool hard_restart = false;

    private:
      DistSamples(const BLACSGrid* g, HSSMatrixMPI<scalar_t>& hss,
                  const opts_t& opts) :
        n_(hss.cols()), g_(g), hss_(&hss),
        rgen_(random::make_random_generator<real_t>
              (opts.random_engine(), opts.random_distribution())) { }

      int n_ = 0, d_ = 0;
      const BLACSGrid* g_ = nullptr;
      const HSSMatrixMPI<scalar_t>* hss_ = nullptr;
      std::unique_ptr<random::RandomGeneratorBase<real_t>> rgen_;

      const dmult_t* Amult_ = nullptr;
      const DistM_t* A_ = nullptr;

      DistM_t R, Sr, Sc, leaf_A;
      DenseM_t sub_R2, sub_Sr2, sub_Sc2, sub_A;
    };

#endif // DOXYGEN_SHOULD_SKIP_THIS

  } // end namespace HSS
} // end namespace strumpack

#endif // DIST_SAMPLES_HPP
