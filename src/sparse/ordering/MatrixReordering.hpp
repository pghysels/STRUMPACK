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
#ifndef MATRIX_REORDERING_HPP
#define MATRIX_REORDERING_HPP

#include <vector>
#include <memory>

#include "StrumpackOptions.hpp"
#include "StrumpackConfig.hpp"
#if defined(STRUMPACK_USE_MPI)
#include "misc/MPIWrapper.hpp"
#endif

namespace strumpack {

  template<typename scalar_t,typename integer_t> class CSRMatrix;
  template<typename scalar_t,typename integer_t> class FrontalMatrix;
  template<typename integer_t> class SeparatorTree;

  template<typename scalar_t,typename integer_t> class MatrixReordering {
    using Opts_t = SPOptions<scalar_t>;
    using CSR_t = CSRMatrix<scalar_t,integer_t>;
    using F_t = FrontalMatrix<scalar_t,integer_t>;

  public:
    MatrixReordering(integer_t  n);

    virtual ~MatrixReordering();

    int nested_dissection(const Opts_t& opts, const CSR_t& A,
                          int nx, int ny, int nz,
                          int components, int width);

    int set_permutation(const Opts_t& opts, const CSR_t& A,
                        const int* p, int base);

    void separator_reordering(const Opts_t& opts, CSR_t& A, F_t* F);

    virtual void clear_tree_data();

    const std::vector<integer_t>& perm() const { return perm_; }
    const std::vector<integer_t>& iperm() const { return iperm_; }

    const SeparatorTree<integer_t>& tree() const { return *sep_tree_; }
    SeparatorTree<integer_t>& tree() { return *sep_tree_; }

  protected:
    virtual void separator_reordering_print
    (integer_t max_nr_neighbours, integer_t max_dim_sep);

    void nested_dissection_print
    (const Opts_t& opts, integer_t nnz, int max_level,
     int total_separators, bool verbose) const;

    std::vector<integer_t> perm_, iperm_;

    std::unique_ptr<SeparatorTree<integer_t>> sep_tree_;

  private:
    void nested_dissection_print
    (const Opts_t& opts, integer_t nnz, bool verbose) const;
  };

} // end namespace strumpack

#endif // MATRIX_REORDERING_HPP
