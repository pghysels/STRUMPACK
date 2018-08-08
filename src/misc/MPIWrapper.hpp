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
#ifndef MPI_WRAPPER_HPP
#define MPI_WRAPPER_HPP
#include <complex>
#include <cassert>
#include <numeric>
#include <memory>
#include "mpi.h"
#include "StrumpackParameters.hpp"

namespace strumpack {

  template<typename T> MPI_Datatype mpi_type();
  template<> inline MPI_Datatype mpi_type<char>() { return MPI_CHAR; }
  template<> inline MPI_Datatype mpi_type<int>() { return MPI_INT; }
  template<> inline MPI_Datatype mpi_type<long>() { return MPI_LONG; }
  template<> inline MPI_Datatype mpi_type<unsigned long>() { return MPI_UNSIGNED_LONG; }
  template<> inline MPI_Datatype mpi_type<long long int>() { return MPI_LONG_LONG_INT; }
  template<> inline MPI_Datatype mpi_type<float>() { return MPI_FLOAT; }
  template<> inline MPI_Datatype mpi_type<double>() { return MPI_DOUBLE; }
  template<> inline MPI_Datatype mpi_type<std::complex<float>>() { return MPI_C_FLOAT_COMPLEX; }
  template<> inline MPI_Datatype mpi_type<std::complex<double>>() { return MPI_C_DOUBLE_COMPLEX; }

  class MPIRequest {
  public:
    MPIRequest() {
      req_ = std::unique_ptr<MPI_Request>(new MPI_Request());
    }
    MPIRequest(const MPIRequest&) = delete;
    MPIRequest(MPIRequest&&) = default;
    MPIRequest& operator=(const MPIRequest&) = delete;
    MPIRequest& operator=(MPIRequest&&) = default;

    void wait() { MPI_Wait(req_.get(), MPI_STATUS_IGNORE); }

  private:
    std::unique_ptr<MPI_Request> req_;
    friend class MPIComm;
  };

  inline void wait_all(std::vector<MPIRequest>& reqs) {
    for (auto& r : reqs) r.wait();
  }

  class MPIComm {
  public:
    MPIComm() {}
    MPIComm(MPI_Comm c) { duplicate(c); }
    MPIComm(const MPIComm& c) { *this = c; }
    MPIComm(MPIComm&& c) { *this = std::move(c); }
    ~MPIComm() {
      if (comm_ != MPI_COMM_NULL)
        MPI_Comm_free(&comm_);
    }
    MPIComm& operator=(const MPIComm& c) {
      if (this != &c) duplicate(c.comm());
      return *this;
    }
    MPIComm& operator=(MPIComm&& c) {
      comm_ = c.comm_;
      c.comm_ = MPI_COMM_NULL;
      return *this;
    }

    MPI_Comm comm() const { return comm_; }

    bool is_null() const { return comm_ == MPI_COMM_NULL; }

    int rank() const {
      assert(comm_ != MPI_COMM_NULL);
      int r;
      MPI_Comm_rank(comm_, &r);
      return r;
    }

    int size() const {
      assert(comm_ != MPI_COMM_NULL);
      int nprocs;
      MPI_Comm_size(comm_, &nprocs);
      return nprocs;
    }

    bool is_root() const { return rank() == 0; }

    void barrier() const { MPI_Barrier(comm_); }

    template<typename T>
    MPIRequest isend(const std::vector<T>& sbuf, int dest, int tag) const {
      MPIRequest req;
      MPI_Isend(sbuf.data(), sbuf.size(), mpi_type<T>(),
                dest, tag, comm_, req.req_.get());
      return std::move(req);
    }

    template<typename T>
    void send(const std::vector<T>& sbuf, int dest, int tag) const {
      MPI_Send(sbuf.data(), sbuf.size(), mpi_type<T>(), dest, tag, comm_);
    }

    template<typename T>
    std::vector<T> recv(int src, int tag) const {
      MPI_Status stat;
      MPI_Probe(src, tag, comm_, &stat);
      int msgsize;
      MPI_Get_count(&stat, mpi_type<T>(), &msgsize);
      std::vector<T> rbuf(msgsize);
      MPI_Recv(rbuf.data(), msgsize, mpi_type<T>(), src, tag,
               comm_, MPI_STATUS_IGNORE);
      return rbuf;
    }

    template<typename T> T all_reduce(T t, MPI_Op op) const {
      MPI_Allreduce(MPI_IN_PLACE, &t, 1, mpi_type<T>(), op, comm_);
      return t;
    }

    template<typename T> T reduce(T t, MPI_Op op) const {
      if (is_root()) MPI_Reduce(MPI_IN_PLACE, &t, 1, mpi_type<T>(), op, 0, comm_);
      else MPI_Reduce(&t, &t, 1, mpi_type<T>(), op, 0, comm_);
      return t;
    }

    template<typename T> void all_reduce(T* t, int ssize, MPI_Op op) const {
      MPI_Allreduce(MPI_IN_PLACE, t, ssize, mpi_type<T>(), op, comm_);
    }

    template<typename T> void reduce(T* t, int ssize, MPI_Op op) const {
      if (is_root()) MPI_Reduce(MPI_IN_PLACE, t, ssize, mpi_type<T>(), op, 0, comm_);
      else MPI_Reduce(t, t, ssize, mpi_type<T>(), op, 0, comm_);
    }

    template<typename T> void all_to_all_v
    (std::vector<std::vector<T>>& sbuf, std::vector<T>& rbuf,
     std::vector<T*>& pbuf) const {
      all_to_all_v(sbuf, rbuf, pbuf, mpi_type<T>());
    }

    template<typename T> void all_to_all_v
    (std::vector<std::vector<T>>& sbuf, std::vector<T>& rbuf,
     std::vector<T*>& pbuf, const MPI_Datatype Ttype) const {
      assert(sbuf.size() == std::size_t(size()));
      auto P = size();
      auto ssizes = new int[4*P];
      auto rsizes = ssizes + P;
      auto sdispl = ssizes + 2*P;
      auto rdispl = ssizes + 3*P;
      for (int p=0; p<P; p++) {
        if (sbuf[p].size() > std::numeric_limits<int>::max()) {
          std::cerr << "# ERROR: 32bit integer overflow in all_to_all_v!!" << std::endl;
          MPI_Abort(comm_, 1);
        }
        ssizes[p] = sbuf[p].size();
      }
      MPI_Alltoall(ssizes, 1, mpi_type<int>(), rsizes, 1, mpi_type<int>(), comm_);
      std::size_t totssize = std::accumulate(ssizes, ssizes+P, 0);
      std::size_t totrsize = std::accumulate(rsizes, rsizes+P, 0);
      if (totrsize > std::numeric_limits<int>::max() ||
          totssize > std::numeric_limits<int>::max()) {
        std::cerr << "# ERROR: 32bit integer overflow in all_to_all_v!!" << std::endl;
        MPI_Abort(comm_, 1);
      }
      T* sendbuf = new T[totssize];
      sdispl[0] = rdispl[0] = 0;
      for (int p=1; p<P; p++) {
        sdispl[p] = sdispl[p-1] + ssizes[p-1];
        rdispl[p] = rdispl[p-1] + rsizes[p-1];
      }
      for (int p=0; p<P; p++)
        std::copy(sbuf[p].begin(), sbuf[p].end(), sendbuf+sdispl[p]);
      std::vector<std::vector<T>>().swap(sbuf);
      rbuf.resize(totrsize);
      MPI_Alltoallv(sendbuf, ssizes, sdispl, Ttype,
                    rbuf.data(), rsizes, rdispl, Ttype, comm_);
      pbuf.resize(P);
      for (int p=0; p<P; p++)
        pbuf[p] = rbuf.data() + rdispl[p];
      delete[] ssizes;
      delete[] sendbuf;
    }

    MPIComm sub(int P0, int P, int stride=1) const {
      if (is_null() || size() == 1)
        return MPIComm(MPI_COMM_NULL);
      assert(P0 + P <= size());
      MPIComm sub_comm;
      std::vector<int> sub_ranks(P);
      for (int i=0; i<P; i++)
        sub_ranks[i] = P0 + i*stride;
      MPI_Group group, sub_group;
      MPI_Comm_group(comm_, &group);                          // get group from comm
      MPI_Group_incl(group, P, sub_ranks.data(), &sub_group); // group ranks [P0,P0+P) into sub_group
      MPI_Comm_create(comm_, sub_group, &sub_comm.comm_);     // create new sub_comm
      MPI_Group_free(&group);
      MPI_Group_free(&sub_group);
      return sub_comm;
    }

    MPIComm sub_self(int p) const {
      if (is_null()) return MPIComm(MPI_COMM_NULL);
      MPIComm c0;
      MPI_Group group, sub_group;
      MPI_Comm_group(comm_, &group);
      MPI_Group_incl(group, 1, &p, &sub_group);
      MPI_Comm_create(comm_, sub_group, &c0.comm_);
      MPI_Group_free(&group);
      MPI_Group_free(&sub_group);
      return c0;
    }

  private:
    MPI_Comm comm_ = MPI_COMM_WORLD;

    void duplicate(MPI_Comm c) {
      if (c == MPI_COMM_NULL) comm_ = c;
      else MPI_Comm_dup(c, &comm_);
    }
  };

  inline int mpi_rank(MPI_Comm c=MPI_COMM_WORLD) {
    assert(c != MPI_COMM_NULL);
    int rank;
    MPI_Comm_rank(c, &rank);
    return rank;
  }

  inline int mpi_nprocs(MPI_Comm c=MPI_COMM_WORLD) {
    assert(c != MPI_COMM_NULL);
    int nprocs;
    MPI_Comm_size(c, &nprocs);
    return nprocs;
  }

  inline bool mpi_root(MPI_Comm c=MPI_COMM_WORLD) {
    int flag;
    MPI_Initialized(&flag);
    if (flag) return mpi_rank(c) == 0;
    else return true;
  }

  inline MPI_Comm mpi_sub_comm(MPI_Comm comm, int P0, int P) {
    if (P == 1 || comm == MPI_COMM_NULL) return MPI_COMM_NULL;
    MPI_Comm sub_comm;
    std::vector<int> sub_ranks(P);
    for (std::size_t i=0; i<sub_ranks.size(); i++) sub_ranks[i] = P0+i;
    MPI_Group group, sub_group;
    MPI_Comm_group(comm, &group);                           // get group from comm
    MPI_Group_incl(group, P, sub_ranks.data(), &sub_group); // group ranks [P0,P0+P) into sub_group
    MPI_Comm_create(comm, sub_group, &sub_comm);            // create new sub_comm
    MPI_Group_free(&group);
    MPI_Group_free(&sub_group);
    return sub_comm;
  }

  inline void mpi_free_comm(MPI_Comm* comm) {
    if (*comm != MPI_COMM_WORLD && *comm != MPI_COMM_NULL) {
      MPI_Comm_free(comm);
      *comm = MPI_COMM_NULL;
    }
  }

  template<class T> struct
  MPIRealType { typedef T value_type; };
  template<class T> struct
  MPIRealType<std::complex<T>>{ typedef T value_type; };

  template<typename T> T Allreduce(T t, MPI_Op op, MPI_Comm comm) {
    MPI_Allreduce(MPI_IN_PLACE, &t, 1, mpi_type<T>(), op, comm);
    return t;
  }

  template<typename scalar_t> typename MPIRealType<scalar_t>::value_type
  nrm2_omp_mpi(int N, scalar_t* X, int incx, MPI_Comm comm) {
    using real_t = typename MPIRealType<scalar_t>::value_type;
    real_t local_nrm(0.), nrm;
    if (incx==1) {
#pragma omp parallel for reduction(+:local_nrm)
      for (int i=0; i<N; i++) local_nrm += std::real(std::conj(X[i])*X[i]);
    } else {
#pragma omp parallel for reduction(+:local_nrm)
      for (int i=0; i<N; i++)
        local_nrm += std::real(std::conj(X[i*incx])*X[i*incx]);
    }
    MPI_Allreduce(&local_nrm, &nrm, 1, mpi_type<real_t>(), MPI_SUM, comm);
    STRUMPACK_FLOPS(static_cast<long long int>(N)*2);
    return std::sqrt(nrm);
  }

  template<typename T> inline void
  all_to_all_v(std::vector<std::vector<T>>& sbuf, T*& recvbuf, T**& pbuf,
               MPI_Comm comm, MPI_Datatype Ttype) {
    int P = sbuf.size();
    auto ssizes = new int[4*P];
    auto rsizes = ssizes + P;
    auto sdispl = ssizes + 2*P;
    auto rdispl = ssizes + 3*P;
    for (int p=0; p<P; p++) {
      if (sbuf[p].size() > std::numeric_limits<int>::max()) {
        std::cerr << "# ERROR: 32bit integer overflow in all_to_all_v!!" << std::endl;
        MPI_Abort(comm, 1);
      }
      ssizes[p] = sbuf[p].size();
    }
    MPI_Alltoall(ssizes, 1, mpi_type<int>(), rsizes, 1, mpi_type<int>(), comm);
    std::size_t totssize = std::accumulate(ssizes, ssizes+P, 0);
    std::size_t totrsize = std::accumulate(rsizes, rsizes+P, 0);
    if (totrsize > std::numeric_limits<int>::max() ||
        totssize > std::numeric_limits<int>::max()) {
      std::cerr << "# ERROR: 32bit integer overflow in all_to_all_v!!" << std::endl;
      MPI_Abort(comm, 1);
    }
    T* sendbuf = new T[totssize];
    sdispl[0] = rdispl[0] = 0;
    for (int p=1; p<P; p++) {
      sdispl[p] = sdispl[p-1] + ssizes[p-1];
      rdispl[p] = rdispl[p-1] + rsizes[p-1];
    }
    for (int p=0; p<P; p++)
      std::copy(sbuf[p].begin(), sbuf[p].end(), sendbuf+sdispl[p]);
    std::vector<std::vector<T>>().swap(sbuf);
    recvbuf = new T[totrsize];
    MPI_Alltoallv(sendbuf, ssizes, sdispl, Ttype,
                  recvbuf, rsizes, rdispl, Ttype, comm);
    pbuf = new T*[P];
    for (int p=0; p<P; p++) pbuf[p] = recvbuf + rdispl[p];
    delete[] ssizes;
    delete[] sendbuf;
  }

  template<typename T> inline void
  all_to_all_v(std::vector<std::vector<T>>& sbuf, T*& recvbuf,
               T**& pbuf, MPI_Comm comm) {
    all_to_all_v(sbuf, recvbuf, pbuf, comm, mpi_type<T>());
  }

  template<typename T> inline void
  all_to_all_v(std::vector<std::vector<T>>& sbuf, T*& recvbuf,
               std::size_t& totrsize, MPI_Comm comm, MPI_Datatype Ttype) {
    int P = sbuf.size();
    auto ssizes = new int[4*P];
    auto rsizes = ssizes + P;
    auto sdispl = ssizes + 2*P;
    auto rdispl = ssizes + 3*P;
    for (int p=0; p<P; p++) {
      if (sbuf[p].size() > std::numeric_limits<int>::max()) {
        std::cerr << "# ERROR: 32bit integer overflow in all_to_all_v!!" << std::endl;
        MPI_Abort(comm, 1);
      }
      ssizes[p] = sbuf[p].size();
    }
    MPI_Alltoall(ssizes, 1, mpi_type<int>(), rsizes, 1, mpi_type<int>(), comm);
    std::size_t totssize = std::accumulate(ssizes, ssizes+P, 0);
    totrsize = std::accumulate(rsizes, rsizes+P, 0);
    if (totrsize > std::numeric_limits<int>::max() ||
        totssize > std::numeric_limits<int>::max()) {
      std::cerr << "# ERROR: 32bit integer overflow in all_to_all_v!!" << std::endl;
      MPI_Abort(comm, 1);
    }
    T* sendbuf = new T[totssize];
    sdispl[0] = rdispl[0] = 0;
    for (int p=1; p<P; p++) {
      sdispl[p] = sdispl[p-1] + ssizes[p-1];
      rdispl[p] = rdispl[p-1] + rsizes[p-1];
    }
    for (int p=0; p<P; p++)
      std::copy(sbuf[p].begin(), sbuf[p].end(), sendbuf+sdispl[p]);
    std::vector<std::vector<T>>().swap(sbuf);
    recvbuf = new T[totrsize];
    MPI_Alltoallv(sendbuf, ssizes, sdispl, Ttype, recvbuf, rsizes,
                  rdispl, Ttype, comm);
    delete[] ssizes;
    delete[] sendbuf;
  }

  template<typename T> inline void
  all_to_all_v(std::vector<std::vector<T>>& sbuf, T*& recvbuf,
               std::size_t& totrsize, MPI_Comm comm) {
    all_to_all_v(sbuf, recvbuf, totrsize, comm, mpi_type<T>());
  }

} // end namespace strumpack

#endif
