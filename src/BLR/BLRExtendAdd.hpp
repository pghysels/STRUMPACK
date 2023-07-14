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
 * five (5) year renewals, the U.S. Government igs granted for itself
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
#ifndef BLR_EXTEND_ADD_HPP
#define BLR_EXTEND_ADD_HPP

#include "BLRMatrix.hpp"
#include "BLRMatrixMPI.hpp"

namespace strumpack {

  // forward declarations
  template<typename scalar_t,typename integer_t> class FrontalMatrix;
  template<typename scalar_t,typename integer_t> class FrontalMatrixMPI;
  template<typename scalar_t,typename integer_t> class FrontalMatrixBLRMPI;

  namespace BLR {

    template<typename scalar_t,typename integer_t> class BLRExtendAdd {
      using DenseM_t = DenseMatrix<scalar_t>;
      using DistM_t = DistributedMatrix<scalar_t>;
      using BLR_t = BLRMatrix<scalar_t>;
      using BLRMPI_t = BLRMatrixMPI<scalar_t>;
      using F_t = FrontalMatrix<scalar_t,integer_t>;
      using FMPI_t = FrontalMatrixMPI<scalar_t,integer_t>;
      using FBLRMPI_t = FrontalMatrixBLRMPI<scalar_t,integer_t>;
      using VI_t = std::vector<std::size_t>;
      using VVS_t = std::vector<std::vector<scalar_t>>;

    public:
      static void
      copy_to_buffers(const DistM_t& CB, VVS_t& sbuf,
                      const FBLRMPI_t* pa, const VI_t& I);
      static void
      copy_to_buffers(const BLRMPI_t& CB, VVS_t& sbuf,
                      const FBLRMPI_t* pa, const VI_t& I);
      static void
      copy_to_buffers_col(const DistM_t& CB, VVS_t& sbuf,
                          const FBLRMPI_t* pa, const VI_t& I,
                          integer_t begin_col, integer_t end_col);
      static void
      copy_to_buffers_col(const BLRMPI_t& CB, VVS_t& sbuf,
                          const FBLRMPI_t* pa, const VI_t& I,
                          integer_t begin_col, integer_t end_col);

      static void
      copy_from_buffers(BLRMPI_t& F11, BLRMPI_t& F12,
                        BLRMPI_t& F21, BLRMPI_t& F22, scalar_t** pbuf,
                        const FBLRMPI_t* pa, const FMPI_t* ch);
      static void
      copy_from_buffers_col(BLRMPI_t& F11, BLRMPI_t& F12,
                            BLRMPI_t& F21, BLRMPI_t& F22, scalar_t** pbuf,
                            const FBLRMPI_t* pa, const FMPI_t* ch,
                            integer_t begin_col, integer_t end_col);
      static void
      copy_from_buffers(BLRMPI_t& F11, BLRMPI_t& F12,
                        BLRMPI_t& F21, BLRMPI_t& F22, scalar_t** pbuf,
                        const FBLRMPI_t* pa, const FBLRMPI_t* ch);
      static void
      copy_from_buffers_col(BLRMPI_t& F11, BLRMPI_t& F12,
                            BLRMPI_t& F21, BLRMPI_t& F22, scalar_t** pbuf,
                            const FBLRMPI_t* pa, const FBLRMPI_t* ch,
                            integer_t begin_col, integer_t end_col);

      static void
      seq_copy_to_buffers(const DenseM_t& CB, VVS_t& sbuf,
                          const FBLRMPI_t* pa, const F_t* ch);
      static void
      seq_copy_to_buffers(const BLR_t& CB, VVS_t& sbuf,
                          const FBLRMPI_t* pa, const F_t* ch);
      static void
      seq_copy_to_buffers_col(const DenseM_t& CB, VVS_t& sbuf,
                              const FBLRMPI_t* pa, const F_t* ch,
                              integer_t begin_col, integer_t end_col);
      static void
      blrseq_copy_to_buffers_col(const BLR_t& CB, VVS_t& sbuf,
                                 const FBLRMPI_t* pa, const F_t* ch,
                                 integer_t begin_col, integer_t end_col,
                                 const BLROptions<scalar_t>& opts);

      static void
      seq_copy_from_buffers(BLRMPI_t& F11, BLRMPI_t& F12,
                            BLRMPI_t& F21, BLRMPI_t& F22, scalar_t*& pbuf,
                            const FBLRMPI_t* pa, const F_t* ch);
      static void
      seq_copy_from_buffers_col(BLRMPI_t& F11, BLRMPI_t& F12,
                                BLRMPI_t& F21, BLRMPI_t& F22, scalar_t*& pbuf,
                                const FBLRMPI_t* pa, const F_t* ch,
                                integer_t begin_col, integer_t end_col);
    };

  } // end namespace BLR
} // end namespace strumpack

#endif // BLR_EXTEND_ADD_HPP
