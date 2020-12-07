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
#ifndef EXTEND_ADD_HPP
#define EXTEND_ADD_HPP

#include "dense/DistributedMatrix.hpp"


namespace strumpack {

  // forward declarations
  template<typename scalar_t,typename integer_t> class FrontalMatrix;
  template<typename scalar_t,typename integer_t> class FrontalMatrixMPI;
  namespace BLR {
    template<typename scalar_t> class BLRMatrixMPI;
  }

  template<typename scalar_t,typename integer_t> class ExtendAdd {
    using DenseM_t = DenseMatrix<scalar_t>;
    using DistM_t = DistributedMatrix<scalar_t>;
    using BLRMPI_t = BLR::BLRMatrixMPI<scalar_t>;
    using F_t = FrontalMatrix<scalar_t,integer_t>;
    using FMPI_t = FrontalMatrixMPI<scalar_t,integer_t>;
    using VI_t = std::vector<std::size_t>;
    using VVS_t = std::vector<std::vector<scalar_t>>;

  public:
    static void extend_add_copy_to_buffers
    (const DistM_t& CB, VVS_t& sbuf, const FMPI_t* pa, const VI_t& I);

    static void extend_add_seq_copy_to_buffers
    (const DenseM_t& CB, VVS_t& sbuf, const FMPI_t* pa, const F_t* ch);

    static void extend_add_seq_copy_from_buffers
    (DistM_t& F11, DistM_t& F12, DistM_t& F21, DistM_t& F22,
     scalar_t*& pbuf, const FMPI_t* pa, const F_t* ch);

    static void extend_add_copy_from_buffers
    (DistM_t& F11, DistM_t& F12, DistM_t& F21, DistM_t& F22,
     scalar_t** pbuf, const FMPI_t* pa, const FMPI_t* ch);

    static void extend_add_column_copy_to_buffers
    (const DistM_t& CB, VVS_t& sbuf, const FMPI_t* pa, const VI_t& I);

    static void extend_add_column_seq_copy_to_buffers
    (const DenseM_t& CB, VVS_t& sbuf, const FMPI_t* pa, const F_t* ch);

    static void extend_add_column_seq_copy_from_buffers
    (DistM_t& b, DistM_t& bupd, scalar_t*& pbuf,
     const FMPI_t* pa, const F_t* ch);

    static void extend_add_column_copy_from_buffers
    (DistM_t& b, DistM_t& bupd, scalar_t** pbuf,
     const FMPI_t* pa, const FMPI_t* ch);

    static void skinny_extend_add_copy_to_buffers
    (const DistM_t& cS, VVS_t& sbuf, const FMPI_t* pa, const VI_t& I);


    /*
     * This does not do the 'extend' part, that was already done at
     * the child. Here, just send to the parent, so it can be added.
     */
    static void skinny_extend_add_seq_copy_to_buffers
    (const DenseM_t& cS, VVS_t& sbuf, const FMPI_t* pa);

    static void skinny_extend_add_copy_from_buffers
    (DistM_t& S, scalar_t** pbuf, const FMPI_t* pa, const FMPI_t* ch);

    static void skinny_extend_add_seq_copy_from_buffers
    (DistM_t& S, scalar_t*& pbuf, const FMPI_t* pa, const F_t* ch);

    /*
     * TODO what if B is not active?? Do we have the correct processor
     * grid info???
     */
    static void extend_copy_to_buffers
    (const DistM_t& F, const VI_t& I, const VI_t& J,
     const DistM_t& B, VVS_t& sbuf);

    static void extend_copy_from_buffers
    (DistM_t& F, const VI_t& oI, const VI_t& oJ, const DistM_t& B,
     std::vector<scalar_t*>& pbuf);

    static void extract_column_copy_to_buffers
    (const DistM_t& b, const DistM_t& bupd, VVS_t& sbuf,
     const FMPI_t* pa, const FMPI_t* ch);

    static void extract_column_seq_copy_to_buffers
    (const DistM_t& b, const DistM_t& bupd, std::vector<scalar_t>& sbuf,
     const FMPI_t* pa, const F_t* ch);

    static void extract_column_copy_from_buffers
    (DistM_t& CB, std::vector<scalar_t*>& pbuf, const FMPI_t* pa,
     const F_t* ch);

    static void extract_column_seq_copy_from_buffers
    (DenseM_t& CB, std::vector<scalar_t*>& pbuf,
     const FMPI_t* pa, const F_t* ch);

    static void extract_copy_to_buffers
    (const DistM_t& F, const VI_t& I, const VI_t& J, const VI_t& oI,
     const VI_t& oJ, const DistM_t& B, VVS_t& sbuf);

    static void extract_copy_from_buffers
    (DistM_t& F, const VI_t& I, const VI_t& J, const VI_t& oI, const VI_t& oJ,
     const DistM_t& B, std::vector<scalar_t*>& pbuf);

    static void extract_copy_to_buffers
    (const BLRMPI_t& F, const VI_t& I, const VI_t& J, const VI_t& oI,
     const VI_t& oJ, const DistM_t& B, VVS_t& sbuf);

    static void extract_copy_from_buffers
    (DistM_t& F, const VI_t& I, const VI_t& J, const VI_t& oI, const VI_t& oJ,
     const BLRMPI_t& B, std::vector<scalar_t*>& pbuf);
  };

  // forward declaration
  template<typename scalar_t,typename integer_t> class CompressedSparseMatrix;

  template<typename scalar_t,typename integer_t> class ExtractFront {
    using CSM = CompressedSparseMatrix<scalar_t,integer_t>;
    using DistM_t = DistributedMatrix<scalar_t>;

  public:
    static void extract_F11
    (DistM_t& F, const CSM& A, integer_t sep_begin, integer_t dim_sep);

    static void extract_F12
    (DistM_t& F, const CSM& A, integer_t upd_row_begin,
     integer_t upd_col_begin, const std::vector<integer_t>& upd);

    static void extract_F21
    (DistM_t& F, const CSM& A, integer_t upd_row_begin,
     integer_t upd_col_begin, const std::vector<integer_t>& upd);
  };

} // end namespace strumpack

#endif // EXTEND_ADD_HPP
