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
   std::vector<integer_t>& dist, MPI_Comm comm) {
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
  public:
    char* data = NULL;
    integer_t nr_sep = 0;
    integer_t* lchild = NULL;
    integer_t* rchild = NULL;
    integer_t root = -1;
    float* work = NULL;
    integer_t* sep_ptr = NULL;
    integer_t* dim_upd = NULL;
    integer_t** upd = NULL;
    std::unordered_map<integer_t, HSS::HSSPartitionTree> sep_HSS_tree;

    // send the symbolic info of the entire tree belonging to dist_sep
    // owned by owner to [P0,P0+P) send only the root of the sub tree
    // to [P0_brother,P0_brother+P_brother)
    template<typename scalar_t> RedistSubTree
    (const MatrixReorderingMPI<scalar_t,integer_t>& nd,
     const std::vector<integer_t>* _upd, float* _work,
     integer_t P0, integer_t P, integer_t P0_sibling,
     integer_t P_sibling, integer_t owner, MPI_Comm comm) {
      auto rank = mpi_rank(comm);
      int dest0 = std::min(P0, P0_sibling);
      int dest1 = std::max(P0+P, P0_sibling+P_sibling);
      char* send_buf = NULL;
      std::vector<MPI_Request> send_req;
      if (rank == owner) {
        auto nbsep = nd.local_sep_tree->separators();
        size_t msg_size = sizeof(integer_t)*(1+nbsep*3+1+1) +
          sizeof(float)*nbsep;
        for (integer_t i=0; i<nbsep; i++)
          msg_size += sizeof(integer_t)*(1+_upd[i].size());
        std::vector<int> hss_partitions =
          serialize(nd.local_sep_tree->HSS_trees());
        msg_size += sizeof(int) * hss_partitions.size();

        send_buf = static_cast<char*>( ::operator new(msg_size) );
        char* p_buf = send_buf;
        *reinterpret_cast<integer_t*>(p_buf) = nbsep;
        p_buf += sizeof(nbsep);
        std::copy
          (nd.local_sep_tree->lch(), nd.local_sep_tree->lch()+nbsep,
           reinterpret_cast<integer_t*>(p_buf));
        p_buf += sizeof(integer_t)*nbsep;
        std::copy
          (nd.local_sep_tree->rch(), nd.local_sep_tree->rch()+nbsep,
           reinterpret_cast<integer_t*>(p_buf));
        p_buf += sizeof(integer_t)*nbsep;
        // send the index of the root
        *reinterpret_cast<integer_t*>(p_buf) = nd.local_sep_tree->root();
        p_buf += sizeof(integer_t);
        std::copy(_work, _work+nbsep, reinterpret_cast<float*>(p_buf));
        p_buf += sizeof(float)*nbsep;
        for (integer_t s=0; s<nbsep+1; s++) {
          *reinterpret_cast<integer_t*>(p_buf) =
            nd.local_sep_tree->sizes()[s] + nd.sub_graph_range.first;
          p_buf += sizeof(integer_t);
        }
        for (integer_t i=0; i<nbsep; i++) {
          *reinterpret_cast<integer_t*>(p_buf) =
            static_cast<integer_t>(_upd[i].size());
          p_buf += sizeof(integer_t);
        }
        for (integer_t i=0; i<nbsep; i++) {
          std::copy
            (_upd[i].begin(), _upd[i].end(),
             reinterpret_cast<integer_t*>(p_buf));
          p_buf += sizeof(integer_t)*_upd[i].size();
        }
        std::copy
          (hss_partitions.begin(), hss_partitions.end(),
           reinterpret_cast<int*>(p_buf));
        p_buf +=  sizeof(int)*hss_partitions.size();
        assert(p_buf == send_buf+msg_size);
        if (msg_size >= std::numeric_limits<int>::max())
          std::cerr << "ERROR: In " << __FILE__ << ", line "
                    << __LINE__ << ",\n"
                    << "\tmessage is more than "
                    << std::numeric_limits<int>::max() << " bytes."
                    << std::endl
                    << "\tPlease send this message to"
                    << " the STRUMPACK developers." << std::endl;
        send_req.resize(dest1-dest0);

        // TODO the sibling only needs the root of the tree!!
        for (int dest=dest0; dest<dest1; dest++)
          MPI_Isend
            (send_buf, msg_size, MPI_BYTE, dest, 0, comm,
             &send_req[dest-dest0]);
      }
      bool receiver = (rank >= dest0 && rank < dest1);
      if (receiver) {
        MPI_Status stat;
        MPI_Probe(owner, 0, comm, &stat);
        int msg_size;
        MPI_Get_count(&stat, MPI_BYTE, &msg_size);
        data = static_cast<char*>( ::operator new(msg_size) );
        MPI_Recv(data, msg_size, MPI_BYTE, owner, 0, comm, &stat);
      }
      if (rank == owner) {
        MPI_Waitall(send_req.size(), send_req.data(), MPI_STATUSES_IGNORE);
        ::operator delete(send_buf);
        send_req.clear();
      }
      if (receiver) {
        char* p_data = data;
        nr_sep = *((integer_t*)p_data);
        p_data += sizeof(integer_t);
        lchild = (integer_t*)p_data;
        p_data += nr_sep * sizeof(integer_t);
        rchild = (integer_t*)p_data;
        p_data += nr_sep * sizeof(integer_t);
        root = *((integer_t*)p_data);
        p_data += sizeof(integer_t);
        work = (float*)p_data;
        p_data += nr_sep * sizeof(float);
        sep_ptr = (integer_t*)p_data;
        p_data += (nr_sep+1) * sizeof(integer_t);
        dim_upd = (integer_t*)p_data;
        p_data += nr_sep * sizeof(integer_t);
        upd = new integer_t*[nr_sep];
        upd[0] = (integer_t*)p_data;
        for (integer_t sep=0; sep<nr_sep-1; sep++) {
          upd[sep+1] = upd[sep] + dim_upd[sep];
          p_data += dim_upd[sep] * sizeof(integer_t);
        }
        p_data += dim_upd[nr_sep-1] * sizeof(integer_t);
        int sep_tree_size = *((int*)p_data);
        p_data += sizeof(int);
        for (int t=0; t<sep_tree_size; t++) {
          int sep = *((int*)p_data);
          p_data += sizeof(int);
          int buf_size = *((int*)p_data);
          p_data += sizeof(int);
          sep_HSS_tree[sep] = HSS::HSSPartitionTree(buf_size, (int*)p_data);
          p_data += buf_size * sizeof(int);
        }
      }
    }

    ~RedistSubTree() {
      ::operator delete(data);
      delete[] upd;
    }
  };

} // end namespace strumpack

#endif // REDISTRIBUTE_HPP
