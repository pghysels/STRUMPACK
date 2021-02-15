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
 *
 */
#include "FrontalMatrixGPUKernels.hpp"


namespace strumpack {
  namespace gpu {


    template<typename T> void
    assemble(unsigned int nf, AssembleData<T>* dat,
             AssembleData<T>* ddat) {
      std::cout << "TODO assemble" << std::endl;
    }

    template<typename T, typename real_t>
    void replace_pivots(int n, T* A, real_t thresh, gpu::Stream& s) {
      if (!n) return;
      std::cout << "TODO replace pivots" << std::endl;
    }

    template<typename T, int NT, typename real_t>
    void factor_block_batch(unsigned int count, FrontData<T>* dat,
                            bool replace, real_t thresh) {
      if (!count) return;
      std::cout << "TODO factor_block_batch" << std::endl;
    }


    template<typename T> void
    extend_add_rhs(int nrhs, unsigned int nf,
                   AssembleData<T>* dat, AssembleData<T>* ddat) {
      std::cout << "TODO extend_add_rhs" << std::endl;
    }

    template<typename T, int NT> void
    fwd_block_batch(int nrhs, unsigned int count,
                    FrontData<T>* dat) {
      if (!count) return;
      std::cout << "TODO fwd_block_batch" << std::endl;
    }


    template<typename T> void
    extract_rhs(int nrhs, unsigned int nf, AssembleData<T>* dat,
                AssembleData<T>* ddat) {
      std::cout << "TODO extract_rhs" << std::endl;
    }

    template<typename T, int NT> void
    bwd_block_batch(int nrhs, unsigned int count,
                    FrontData<T>* dat) {
      if (!count) return;
      std::cout << "TODO bwd_block_batch" << std::endl;
    }

    // explicit template instantiations
    template void assemble(unsigned int, AssembleData<float>*, AssembleData<float>*);
    template void assemble(unsigned int, AssembleData<double>*, AssembleData<double>*);
    template void assemble(unsigned int, AssembleData<std::complex<float>>*, AssembleData<std::complex<float>>*);
    template void assemble(unsigned int, AssembleData<std::complex<double>>*, AssembleData<std::complex<double>>*);

    template void extend_add_rhs(int, unsigned int, AssembleData<float>*, AssembleData<float>*);
    template void extend_add_rhs(int, unsigned int, AssembleData<double>*, AssembleData<double>*);
    template void extend_add_rhs(int, unsigned int, AssembleData<std::complex<float>>*, AssembleData<std::complex<float>>*);
    template void extend_add_rhs(int, unsigned int, AssembleData<std::complex<double>>*, AssembleData<std::complex<double>>*);

    template void extract_rhs(int, unsigned int, AssembleData<float>*, AssembleData<float>*);
    template void extract_rhs(int, unsigned int, AssembleData<double>*, AssembleData<double>*);
    template void extract_rhs(int, unsigned int, AssembleData<std::complex<float>>*, AssembleData<std::complex<float>>*);
    template void extract_rhs(int, unsigned int, AssembleData<std::complex<double>>*, AssembleData<std::complex<double>>*);


    template void factor_block_batch<float,8,float>(unsigned int, FrontData<float>*, bool, float);
    template void factor_block_batch<double,8,double>(unsigned int, FrontData<double>*, bool, double);
    template void factor_block_batch<std::complex<float>,8,float>(unsigned int, FrontData<std::complex<float>>*, bool, float);
    template void factor_block_batch<std::complex<double>,8,double>(unsigned int, FrontData<std::complex<double>>*, bool, double);

    template void factor_block_batch<float,16,float>(unsigned int, FrontData<float>*, bool, float);
    template void factor_block_batch<double,16,double>(unsigned int, FrontData<double>*, bool, double);
    template void factor_block_batch<std::complex<float>,16,float>(unsigned int, FrontData<std::complex<float>>*, bool, float);
    template void factor_block_batch<std::complex<double>,16,double>(unsigned int, FrontData<std::complex<double>>*, bool, double);

    template void factor_block_batch<float,24,float>(unsigned int, FrontData<float>*, bool, float);
    template void factor_block_batch<double,24,double>(unsigned int, FrontData<double>*, bool, double);
    template void factor_block_batch<std::complex<float>,24,float>(unsigned int, FrontData<std::complex<float>>*, bool, float);
    template void factor_block_batch<std::complex<double>,24,double>(unsigned int, FrontData<std::complex<double>>*, bool, double);

    template void factor_block_batch<float,32,float>(unsigned int, FrontData<float>*, bool, float);
    template void factor_block_batch<double,32,double>(unsigned int, FrontData<double>*, bool, double);
    template void factor_block_batch<std::complex<float>,32,float>(unsigned int, FrontData<std::complex<float>>*, bool, float);
    template void factor_block_batch<std::complex<double>,32,double>(unsigned int, FrontData<std::complex<double>>*, bool, double);

    template void replace_pivots(int, float*, float, gpu::Stream&);
    template void replace_pivots(int, double*, double, gpu::Stream&);
    template void replace_pivots(int, std::complex<float>*, float, gpu::Stream&);
    template void replace_pivots(int, std::complex<double>*, double, gpu::Stream&);

    template void fwd_block_batch<float,8>(int, unsigned int, FrontData<float>*);
    template void fwd_block_batch<double,8>(int, unsigned int, FrontData<double>*);
    template void fwd_block_batch<std::complex<float>,8>(int, unsigned int, FrontData<std::complex<float>>*);
    template void fwd_block_batch<std::complex<double>,8>(int, unsigned int, FrontData<std::complex<double>>*);

    template void fwd_block_batch<float,16>(int, unsigned int, FrontData<float>*);
    template void fwd_block_batch<double,16>(int, unsigned int, FrontData<double>*);
    template void fwd_block_batch<std::complex<float>,16>(int, unsigned int, FrontData<std::complex<float>>*);
    template void fwd_block_batch<std::complex<double>,16>(int, unsigned int, FrontData<std::complex<double>>*);

    template void fwd_block_batch<float,24>(int, unsigned int, FrontData<float>*);
    template void fwd_block_batch<double,24>(int, unsigned int, FrontData<double>*);
    template void fwd_block_batch<std::complex<float>,24>(int, unsigned int, FrontData<std::complex<float>>*);
    template void fwd_block_batch<std::complex<double>,24>(int, unsigned int, FrontData<std::complex<double>>*);

    template void fwd_block_batch<float,32>(int, unsigned int, FrontData<float>*);
    template void fwd_block_batch<double,32>(int, unsigned int, FrontData<double>*);
    template void fwd_block_batch<std::complex<float>,32>(int, unsigned int, FrontData<std::complex<float>>*);
    template void fwd_block_batch<std::complex<double>,32>(int, unsigned int, FrontData<std::complex<double>>*);


    template void bwd_block_batch<float,8>(int, unsigned int, FrontData<float>*);
    template void bwd_block_batch<double,8>(int, unsigned int, FrontData<double>*);
    template void bwd_block_batch<std::complex<float>,8>(int, unsigned int, FrontData<std::complex<float>>*);
    template void bwd_block_batch<std::complex<double>,8>(int, unsigned int, FrontData<std::complex<double>>*);

    template void bwd_block_batch<float,16>(int, unsigned int, FrontData<float>*);
    template void bwd_block_batch<double,16>(int, unsigned int, FrontData<double>*);
    template void bwd_block_batch<std::complex<float>,16>(int, unsigned int, FrontData<std::complex<float>>*);
    template void bwd_block_batch<std::complex<double>,16>(int, unsigned int, FrontData<std::complex<double>>*);

    template void bwd_block_batch<float,24>(int, unsigned int, FrontData<float>*);
    template void bwd_block_batch<double,24>(int, unsigned int, FrontData<double>*);
    template void bwd_block_batch<std::complex<float>,24>(int, unsigned int, FrontData<std::complex<float>>*);
    template void bwd_block_batch<std::complex<double>,24>(int, unsigned int, FrontData<std::complex<double>>*);

    template void bwd_block_batch<float,32>(int, unsigned int, FrontData<float>*);
    template void bwd_block_batch<double,32>(int, unsigned int, FrontData<double>*);
    template void bwd_block_batch<std::complex<float>,32>(int, unsigned int, FrontData<std::complex<float>>*);
    template void bwd_block_batch<std::complex<double>,32>(int, unsigned int, FrontData<std::complex<double>>*);

  } // end namespace gpu
} // end namespace strumpack
