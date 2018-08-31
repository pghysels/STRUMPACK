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
#include "MatrixReorderingMPI.hpp"
#include "../misc/MPIWrapper.hpp"

namespace strumpack {

  /**
   * Permute a vector which is distributed over a number of processes
   * according to dist.
   *  - x        the local vector with [dist[p],dist[p+1]) elements
   *  - iorder   the GLOBAL inverse permutation
   *  - dist     describes distribution of the vector, dist has P+1 elements
   *             process p has elements [dist[p],dist[p+1])
   *  - comm     the MPI communicator, mpi_nprocs(comm)==P
   */
  template<typename scalar_t,typename integer_t> void permute_vector
  (scalar_t* x, std::vector<integer_t>& iorder,
   const std::vector<integer_t>& dist, MPI_Comm comm) {
    auto rank = mpi_rank(comm);
    auto P = mpi_nprocs(comm);
    auto lo = dist[rank];
    auto hi = dist[rank+1];
    auto n = hi - lo;
    struct IdxVal { integer_t idx; scalar_t val; };
    auto dest = new int[n+4*P];
    auto scnts = dest+n;
    auto rcnts = scnts+P;
    auto sdispls = rcnts+P;
    auto rdispls = sdispls+P;
    auto sbuf = new IdxVal[n];
    auto pp = new IdxVal*[P];
    std::fill(scnts, scnts+P, 0);
    for (integer_t r=0; r<n; r++) {
      dest[r] = std::upper_bound
        (dist.begin(), dist.end(), iorder[r+lo]) - dist.begin() - 1;
      scnts[dest[r]]++;
    }
    sdispls[0] = 0;
    pp[0] = sbuf;
    for (integer_t p=1; p<P; p++) {
      sdispls[p] = sdispls[p-1] + scnts[p-1];
      pp[p] = sbuf + sdispls[p];
    }
    for (integer_t r=0; r<n; r++) {
      auto p = dest[r];
      pp[p]->idx = iorder[r+lo];
      pp[p]->val = x[r];
      pp[p]++;
    }
    delete[] pp;
    for (integer_t p=0; p<P; p++) {
      sdispls[p] *= sizeof(IdxVal); // convert to bytes
      scnts[p] *= sizeof(IdxVal);
    }
    MPI_Alltoall(scnts, 1, mpi_type<int>(), rcnts, 1, mpi_type<int>(), comm);
    rdispls[0] = 0;
    for (int p=1; p<P; p++)
      rdispls[p] = rdispls[p-1] + rcnts[p-1];
    IdxVal* rbuf = new IdxVal[n];
    MPI_Alltoallv
      (sbuf, scnts, sdispls, MPI_BYTE, rbuf, rcnts, rdispls, MPI_BYTE, comm);
    delete[] dest;
    delete[] sbuf;
#pragma omp parallel for
    for (integer_t i=0; i<n; i++)
      x[rbuf[i].idx-lo] = rbuf[i].val;
    delete[] rbuf;
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
    std::unordered_map<integer_t, HSS::HSSPartitionTree> HSS_tree;
    std::unordered_map<integer_t, std::vector<bool>> admissibility;

    // send the symbolic info of the entire tree belonging to dist_sep
    // owned by owner to [P0,P0+P) send only the root of the sub tree
    // to [P0_brother,P0_brother+P_brother)
    template<typename scalar_t> RedistSubTree
    (const MatrixReorderingMPI<scalar_t,integer_t>& nd,
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
        auto nbsep = nd.local_tree().separators();
        const auto& trees = nd.local_tree().HSS_trees();
        const auto& adm = nd.local_tree().admissibilities();

        sbufi.push_back(nbsep);
        sbufi.insert(sbufi.end(), nd.local_tree().lch(), nd.local_tree().lch()+nbsep);
        sbufi.insert(sbufi.end(), nd.local_tree().rch(), nd.local_tree().rch()+nbsep);
        sbufi.push_back(nd.local_tree().root());
        for (integer_t s=0; s<nbsep+1; s++)
          sbufi.push_back(nd.local_tree().sizes(s) + nd.sub_graph_range.first);
        for (integer_t i=0; i<nbsep; i++)
          sbufi.push_back(_upd[i].size());
        for (integer_t i=0; i<nbsep; i++)
          sbufi.insert(sbufi.end(), _upd[i].begin(), _upd[i].end());
        sbufi.push_back(trees.size());
        for (auto& t : trees) {
          sbufi.push_back(t.first);
          auto ht_buf = t.second.serialize();
          sbufi.push_back(ht_buf.size());
          sbufi.insert(sbufi.end(), ht_buf.begin(), ht_buf.end());
        }
        sbufi.push_back(adm.size());
        for (auto& a : adm) {
          sbufi.push_back(a.first);
          sbufi.push_back(a.second.size());
          sbufi.insert(sbufi.end(), a.second.begin(), a.second.end());
        }

        sbuff.reserve(nbsep);
        sbuff.insert(sbuff.end(), _work.begin(), _work.end());

        if (sbufi.size() >= std::numeric_limits<int>::max())
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
        auto pi = rbufi.begin();
        nr_sep = *pi++;
        lchild = &*pi;   pi += nr_sep;
        rchild = &*pi;   pi += nr_sep;
        root = *pi++;
        sep_ptr = &*pi;  pi += nr_sep + 1;
        dim_upd = &*pi;  pi += nr_sep;
        upd = std::vector<integer_t*>(nr_sep);
        upd[0] = &*pi;   pi += dim_upd[0];
        for (integer_t sep=0; sep<nr_sep-1; sep++) {
          upd[sep+1] = upd[sep] + dim_upd[sep];
          pi += dim_upd[sep+1];
        }
        auto nr_trees = *pi++;
        HSS_tree.reserve(nr_trees);
        for (integer_t t=0; t<nr_trees; t++) {
          auto sep = *pi++;
          auto size = *pi++;
          std::vector<int> hss(size);
          std::copy(pi, pi+size, hss.begin());
          pi += size;
          HSS_tree[sep] = std::move(HSS::HSSPartitionTree(hss));
        }
        auto nr_adm = *pi++;
        admissibility.reserve(nr_adm);
        for (integer_t a=0; a<nr_adm; a++) {
          auto sep = *pi++;
          auto size = *pi++;
          std::vector<bool> adm(size);
          std::copy(pi, pi+size, adm.begin());
          admissibility[sep] = std::move(adm);
        }
        work = rbuff.data();
      }
    }
  };

} // end namespace strumpack

#endif // REDISTRIBUTE_HPP
