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
#ifndef REDISTRIBUTE_HPP
#define REDISTRIBUTE_HPP

#include <limits>
#include <algorithm>

#include "dense/DenseMatrix.hpp"
#include "SeparatorTree.hpp"
#include "misc/MPIWrapper.hpp"
#include "misc/Triplet.hpp"

namespace strumpack {

  /**
   * Permute a vector which is distributed over a number of processes
   * according to dist. This is the special case for a single
   * colom. There is a more general version that can handle multiple
   * columns, but that code does 2 all-to-all calls.
   *
   * \param x Local vector with [dist[p],dist[p+1]) elements.
   * \param iorder The global inverse permutation, size dist[P].
   * \param dist Describes distribution of the vector, dist has P+1
   * elements process p has elements [dist[p],dist[p+1])
   * \param comm The MPI communicator, comm.size() == P
   */
  template<typename scalar_t,typename integer_t> void permute_vector
  (scalar_t* x, std::vector<integer_t>& iorder,
   const std::vector<integer_t>& dist, const MPIComm& comm) {
    auto rank = comm.rank();
    auto P = comm.size();
    auto lo = dist[rank];
    auto m = dist[rank+1] - lo;
    using IdxVal = IdxVal<scalar_t,integer_t>;
    std::vector<int> scnts(P), dest(m);
    for (integer_t r=0; r<m; r++) {
      dest[r] = std::upper_bound
        (dist.begin(), dist.end(), iorder[r+lo]) - dist.begin() - 1;
      scnts[dest[r]]++;
    }
    std::vector<std::vector<IdxVal>> sbuf(P);
    for (int p=0; p<P; p++)
      sbuf[p].reserve(scnts[p]);
    for (integer_t r=0; r<m; r++)
      sbuf[dest[r]].emplace_back(iorder[r+lo], x[r]);
    auto rbuf = comm.all_to_all_v(sbuf);
#pragma omp parallel for
    for (integer_t i=0; i<m; i++)
      x[rbuf[i].i-lo] = rbuf[i].v;
    IdxVal::free_mpi_type();
  }

  /**
   * Permute a vector or multiple vectors, which are distributed over
   * a number of processes according to dist. There is a special case
   * for a single colom, but this is a more general version that can
   * handle multiple columns but does 2 all-to-all calls.
   *
   * \param x Local vectors with [dist[p],dist[p+1]) elements.
   * \param iorder The global inverse permutation, size dist[P].
   * \param dist Describes distribution of the vector, dist has P+1
   * elements process p has elements [dist[p],dist[p+1])
   * \param comm The MPI communicator, comm.size() == P
   */
  template<typename scalar_t,typename integer_t> void permute_vector
  (DenseMatrix<scalar_t>& x, std::vector<integer_t>& iorder,
   const std::vector<integer_t>& dist, const MPIComm& comm) {
    if (x.cols() == 1)
      permute_vector(x.data(), iorder, dist, comm);
    else {
      auto rank = comm.rank();
      auto P = comm.size();
      auto lo = dist[rank];
      auto m = dist[rank+1] - lo;
      integer_t n = x.cols();
      assert(m == integer_t(x.rows()));
      std::vector<int> scnts(P), dest(m);
      for (integer_t r=0; r<m; r++) {
        dest[r] = std::upper_bound
          (dist.begin(), dist.end(), iorder[r+lo]) - dist.begin() - 1;
        scnts[dest[r]]++;
      }
      std::vector<std::vector<scalar_t>> ssbuf(P);
      std::vector<std::vector<integer_t>> isbuf(P);
      for (int p=0; p<P; p++) {
        ssbuf[p].reserve(scnts[p]);
        isbuf[p].reserve(scnts[p]);
      }
      for (integer_t r=0; r<m; r++) {
        for (integer_t c=0; c<n; c++)
          ssbuf[dest[r]].push_back(x(r,c));
        isbuf[dest[r]].push_back(iorder[r+lo]);
      }
      auto srbuf = comm.all_to_all_v(ssbuf);
      auto irbuf = comm.all_to_all_v(isbuf);
#pragma omp parallel for
      for (integer_t r=0; r<m; r++)
        for (integer_t c=0; c<n; c++)
          x(irbuf[r]-lo,c) = srbuf[r*n+c];
    }
  }

  /**
   * Helper class to receive the sub graph that is owned by process
   * owner.  This class will store the separator tree information,
   * like lchild, rchild, sep_ptr, work, ...  for the local subgraph
   * that was originally at process owner.
   */
  template<typename integer_t> class RedistSubTree {
  private:
    std::vector<integer_t> rbufi;
    std::vector<float> rbuff;

  public:
    integer_t nr_sep = 0;
    integer_t* lchild = nullptr;
    integer_t* rchild = nullptr;
    integer_t root = -1;
    integer_t* sep_ptr = nullptr;
    integer_t* dim_upd = nullptr;
    std::vector<integer_t*> upd;
    float* work = nullptr;

    // send the symbolic info of the entire tree belonging to dist_sep
    // owned by owner to [P0,P0+P) send only the root of the sub tree
    // to [P0_brother,P0_brother+P_brother)
    RedistSubTree(const SeparatorTree<integer_t>& tree, integer_t sub_begin,
                  const std::vector<std::vector<integer_t>>& _upd,
                  const std::vector<float>& _work,
                  integer_t P0, integer_t P, integer_t P0_sibling,
                  integer_t P_sibling, integer_t owner, const MPIComm& comm) {
      auto rank = comm.rank();
      int dest0 = std::min(P0, P0_sibling);
      int dest1 = std::max(P0+P, P0_sibling+P_sibling);
      std::vector<integer_t> sbufi;
      std::vector<float> sbuff;
      std::vector<MPIRequest> sreq;
      if (rank == owner) {
        auto nbsep = tree.separators();
        std::size_t sbufi_size = 3 + 4*nbsep;
        for (integer_t i=0; i<nbsep; i++)
          sbufi_size += _upd[i].size();
        sbufi.reserve(sbufi_size);
        sbufi.push_back(nbsep);
        sbufi.insert(sbufi.end(), tree.lch(), tree.lch()+nbsep);
        sbufi.insert(sbufi.end(), tree.rch(), tree.rch()+nbsep);
        sbufi.push_back(tree.root());
        for (integer_t s=0; s<nbsep+1; s++)
          sbufi.push_back(tree.sizes(s) + sub_begin);
        for (integer_t i=0; i<nbsep; i++)
          sbufi.push_back(_upd[i].size());
        for (integer_t i=0; i<nbsep; i++)
          sbufi.insert(sbufi.end(), _upd[i].begin(), _upd[i].end());
        sbuff.reserve(_work.size());
        sbuff.insert(sbuff.end(), _work.begin(), _work.end());
        if (sbufi.size() >=
            static_cast<std::size_t>(std::numeric_limits<int>::max()))
          std::cerr << "ERROR: In " << __FILE__ << ", line "
                    << __LINE__ << ",\n"
                    << "\tmessage is more than "
                    << std::numeric_limits<int>::max() << " bytes."
                    << std::endl
                    << "\tPlease send this message to"
                    << " the STRUMPACK developers." << std::endl;
        sreq.reserve(2*(dest1-dest0));
        // TODO the sibling only needs the root of the tree!!
        for (int dest=dest0; dest<dest1; dest++) {
          sreq.emplace_back(comm.isend(sbufi, dest, 0));
          sreq.emplace_back(comm.isend(sbuff, dest, 1));
        }
      }
      bool receiver = (rank >= dest0 && rank < dest1);
      if (receiver) {
        rbufi = comm.template recv<integer_t>(owner, 0);
        rbuff = comm.template recv<float>(owner, 1);
      }
      if (rank == owner) {
        wait_all(sreq);
        sbuff.clear(); sbuff.shrink_to_fit();
        sbufi.clear(); sbufi.shrink_to_fit();
      }
      if (receiver) {
        auto pi = rbufi.data();
        nr_sep = *pi++;
        if (nr_sep) {
          lchild = pi;   pi += nr_sep;
          rchild = pi;   pi += nr_sep;
          root = *pi++;
          sep_ptr = pi;  pi += nr_sep + 1;
          dim_upd = pi;  pi += nr_sep;
          upd = std::vector<integer_t*>(nr_sep);
          upd[0] = &*pi;   pi += dim_upd[0];
          for (integer_t sep=0; sep<nr_sep-1; sep++) {
            upd[sep+1] = upd[sep] + dim_upd[sep];
            pi += dim_upd[sep+1];
          }
        }
        work = rbuff.data();
      }
    }
  };

} // end namespace strumpack

#endif // REDISTRIBUTE_HPP
