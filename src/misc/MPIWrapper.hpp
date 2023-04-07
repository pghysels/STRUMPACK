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
/*!
 * \file MPIWrapper.hpp
 * \brief Contains some simple C++ MPI wrapper utilities.
 */
#ifndef STRUMPACK_MPI_WRAPPER_HPP
#define STRUMPACK_MPI_WRAPPER_HPP

#include <list>
#include <vector>
#include <complex>
#include <cassert>
#include <numeric>
#include <limits>
#include <memory>
#include <utility>
#include "StrumpackConfig.hpp"

#define OMPI_SKIP_MPICXX 1
#include <mpi.h>

#include "StrumpackParameters.hpp"
#include "Triplet.hpp"

namespace strumpack {

  /**
   * Return the corresponding MPI_Datatype, for simple C++ data types.
   *
   * \tparam T C++ type for which to return the corresponding
   * MPI_Datatype
   */
  template<typename T> MPI_Datatype mpi_type() { return T::mpi_type(); }
  /** return MPI datatype for C++ char */
  template<> inline MPI_Datatype mpi_type<char>() { return MPI_CHAR; }
  /** return MPI datatype for C++ bool */
  //template<> inline MPI_Datatype mpi_type<bool>() { return MPI_CXX_BOOL; }
  template<> inline MPI_Datatype mpi_type<bool>() { return MPI_CHAR; }
  /** return MPI datatype for C++ int */
  template<> inline MPI_Datatype mpi_type<int>() { return MPI_INT; }
  /** return MPI datatype for C++ long */
  template<> inline MPI_Datatype mpi_type<long>() { return MPI_LONG; }
  /** return MPI datatype for C++ unsigned long */
  template<> inline MPI_Datatype mpi_type<unsigned long>() { return MPI_UNSIGNED_LONG; }
  /** return MPI datatype for C++ long long int */
  template<> inline MPI_Datatype mpi_type<long long int>() { return MPI_LONG_LONG_INT; }
  /** return MPI datatype for C++ float */
  template<> inline MPI_Datatype mpi_type<float>() { return MPI_FLOAT; }
  /** return MPI datatype for C++ double */
  template<> inline MPI_Datatype mpi_type<double>() { return MPI_DOUBLE; }
  /** return MPI datatype for C++ std::complex<float> */
  //template<> inline MPI_Datatype mpi_type<std::complex<float>>() { return MPI_CXX_FLOAT_COMPLEX; }
  template<> inline MPI_Datatype mpi_type<std::complex<float>>() { return MPI_C_FLOAT_COMPLEX; }
  /** return MPI datatype for C++ std::complex<double> */
  //template<> inline MPI_Datatype mpi_type<std::complex<double>>() { return MPI_CXX_DOUBLE_COMPLEX; }
  template<> inline MPI_Datatype mpi_type<std::complex<double>>() { return MPI_C_DOUBLE_COMPLEX; }
  /** return MPI datatype for C++ std::pair<int,int> */
  template<> inline MPI_Datatype mpi_type<std::pair<int,int>>() { return MPI_2INT; }

  /** return MPI datatype for C++ std::pair<long int,long int> */
  template<> inline MPI_Datatype mpi_type<std::pair<long int,long int>>() {
    static MPI_Datatype l_l_mpi_type = MPI_DATATYPE_NULL;
    if (l_l_mpi_type == MPI_DATATYPE_NULL) {
      MPI_Type_contiguous
        (2, strumpack::mpi_type<long int>(), &l_l_mpi_type);
      MPI_Type_commit(&l_l_mpi_type);
    }
    return l_l_mpi_type;
  }
  /** return MPI datatype for C++ std::pair<long long int,long long int> */
  template<> inline MPI_Datatype mpi_type<std::pair<long long int,long long int>>() {
    static MPI_Datatype ll_ll_mpi_type = MPI_DATATYPE_NULL;
    if (ll_ll_mpi_type == MPI_DATATYPE_NULL) {
      MPI_Type_contiguous
        (2, strumpack::mpi_type<long long int>(), &ll_ll_mpi_type);
      MPI_Type_commit(&ll_ll_mpi_type);
    }
    return ll_ll_mpi_type;
  }

  /**
   * \class MPIRequest
   * \brief Wrapper around an MPI_Request object.
   *
   * This is a very basic wrapper around an MPI_Request object. Since
   * MPI calls take a pointer to the MPI_Request object, it does not
   * make sense to copy an MPI_Request object. The MPIRequest object
   * can be moved though.
   *
   * One problem with this class is that it cannot be used to create a
   * vector of MPI_Request objects to be used in for instance an
   * MPI_Waitall or MPI_Waitany call. For that you should use
   * MPI_Request manually.
   *
   * \see wait_all
   */
  class MPIRequest {
  public:
    /**
     * Default constructor, construct an empty MPIRequest.
     */
    MPIRequest() {
      req_ = std::unique_ptr<MPI_Request>(new MPI_Request());
    }

    /**
     * Copy constructor is not available. Copying an MPI_Request is
     * not allowed.
     */
    MPIRequest(const MPIRequest&) = delete;

    /**
     * Default move constructor.
     */
    MPIRequest(MPIRequest&&) = default;

    /**
     * Copy assignment is disabled.
     */
    MPIRequest& operator=(const MPIRequest&) = delete;

    /**
     * Default move assignment.
     */
    MPIRequest& operator=(MPIRequest&&) = default;

    /**
     * Wait for the request to complete.
     */
    void wait() { MPI_Wait(req_.get(), MPI_STATUS_IGNORE); }

  private:
    std::unique_ptr<MPI_Request> req_;
    friend class MPIComm;
  };

  /**
   * Wait on all MPIRequests in a vector. Note that the MPI_Requests
   * are not stored contiguously, and hence the implementation of this
   * routine cannot use MPI_Waitall, but must wait on all requests
   * individually.
   *
   * If you need MPI_Waitall (or MPI_Waitany), for performance
   * reasons, you should use a vector<MPI_Request>.
   */
  inline void wait_all(std::vector<MPIRequest>& reqs) {
    for (auto& r : reqs) r.wait();
    reqs.clear();
  }

  inline void wait_all(std::vector<MPI_Request>& reqs) {
    MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
  }

  /**
   * \class MPIComm
   * \brief Wrapper class around an MPI_Comm object.
   *
   * This is a simple wrapper around a MPI_Comm object. The main
   * reason for this class is to simplify resource management of the
   * MPI_Comm object. An object of class MPIComm owns the MPI_Comm
   * that it stores, and is responsible for freeing it (in the
   * destructor).
   *
   * A number of simple wrappers around basic MPI calls are provided.
   */
  class MPIComm {
  public:
    /**
     * Default constructor. This will initialize the encapsulated
     * MPI_Comm to MPI_COMM_WORLD.
     */
    MPIComm() {}

    /**
     * Constructor using a MPI_Comm. This will DUPLICATE the input
     * communicator c!
     *
     * \param c the input MPI communicator, this will be duplicated
     * internally and can thus be freed immediately
     */
    MPIComm(MPI_Comm c) { duplicate(c); }

    /**
     * Copy constructor. Will DUPLICATE the underlying MPI_Comm
     * object!
     *
     * \param c will be copied, not changed
     */
    MPIComm(const MPIComm& c) { *this = c; }

    /**
     * Move constructor.
     *
     * \param c will be moved from, will be reset to MPI_COMM_NULL
     * \see operator=(MPIComm&& c)
     */
    MPIComm(MPIComm&& c) noexcept { *this = std::move(c); }

    /**
     * Virtual destructor.  Free the MPI_Comm object, unless it is
     * MPI_COMM_NULL or MPI_COMM_WORLD.
     */
    virtual ~MPIComm() {
      if (comm_ != MPI_COMM_NULL && comm_ != MPI_COMM_WORLD)
        MPI_Comm_free(&comm_);
    }

    /**
     * Assignement operator. This will DUPLICATE the MPI_Comm object.
     *
     * \param c object to copy from, will not be modified.
     */
    MPIComm& operator=(const MPIComm& c) {
      if (this != &c) duplicate(c.comm());
      return *this;
    }

    /**
     * Move operator.
     *
     * \param c the object ro be moved from, will be reset to
     * MPI_COMM_NULL
     */
    MPIComm& operator=(MPIComm&& c) noexcept {
      comm_ = c.comm_;
      c.comm_ = MPI_COMM_NULL;
      return *this;
    }

    /**
     * Returns the underlying MPI_Comm object.
     */
    MPI_Comm comm() const { return comm_; }

    /**
     * Checks whether the underlying MPI_Comm object is MPI_COMM_NULL.
     */
    bool is_null() const { return comm_ == MPI_COMM_NULL; }

    /**
     * Return the current rank in this communicator.
     */
    int rank() const {
      assert(comm_ != MPI_COMM_NULL);
      int r;
      MPI_Comm_rank(comm_, &r);
      return r;
    }

    /**
     * Return the size of, ie, number of processes in, this
     * communicator.  This communicator should not be MPI_COMM_NULL.
     */
    int size() const {
      assert(comm_ != MPI_COMM_NULL);
      int nprocs;
      MPI_Comm_size(comm_, &nprocs);
      return nprocs;
    }

    /**
     * Check whether the current process is the root of this MPI
     * communicator.
     */
    bool is_root() const { return rank() == 0; }

    /**
     * Perform a barrier operation. This operation is collective on
     * all the ranks in this communicator.
     */
    void barrier() const { MPI_Barrier(comm_); }

    template<typename T> void
    broadcast(std::vector<T>& sbuf) const {
      MPI_Bcast(sbuf.data(), sbuf.size(), mpi_type<T>(), 0, comm_);
    }

    template<typename T> void
    broadcast_from(std::vector<T>& sbuf, int src) const {
      MPI_Bcast(sbuf.data(), sbuf.size(), mpi_type<T>(), src, comm_);
    }

    template<typename T, std::size_t N> void
    broadcast(std::array<T,N>& sbuf) const {
      MPI_Bcast(sbuf.data(), sbuf.size(), mpi_type<T>(), 0, comm_);
    }

    template<typename T> void broadcast(T& data) const {
      MPI_Bcast(&data, 1, mpi_type<T>(), 0, comm_);
    }
    template<typename T> void broadcast_from(T& data, int src) const {
      MPI_Bcast(&data, 1, mpi_type<T>(), src, comm_);
    }
    template<typename T> void
    broadcast(T* sbuf, std::size_t ssize) const {
      MPI_Bcast(sbuf, ssize, mpi_type<T>(), 0, comm_);
    }
    template<typename T> void
    broadcast_from(T* sbuf, std::size_t ssize, int src) const {
      MPI_Bcast(sbuf, ssize, mpi_type<T>(), src, comm_);
    }

    template<typename T>
    void all_gather(T* buf, std::size_t rsize) const {
      MPI_Allgather
        (MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
         buf, rsize, mpi_type<T>(), comm_);
    }

    template<typename T>
    void all_gather_v(T* buf, const int* rcnts, const int* displs) const {
      MPI_Allgatherv
        (MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, buf, rcnts, displs,
         mpi_type<T>(), comm_);
    }

    template<typename T>
    void gather(T* sbuf, int ssize, int* rbuf, int rsize, int root) const {
      MPI_Gather
        (sbuf, ssize, mpi_type<T>(), rbuf,
         rsize, mpi_type<T>(), root, comm_);
    }

    template<typename T>
    void gather_v(T* sbuf, int scnts, T* rbuf, const int* rcnts,
                  const int* displs, int root) const {
      MPI_Gatherv
        (sbuf, scnts, mpi_type<T>(), rbuf, rcnts, displs,
         mpi_type<T>(), root, comm_);
    }


    /**
     * Non-blocking send of a vector to a destination process, with a
     * certain tag.
     *
     * \tparam T template type of the send buffer, should have a
     * corresponding mpi_type<T>() implementation
     *
     * \param sbuf buffer of type T to be send
     * \param dest rank of destination process in this MPI communicator
     * \param tag tag to use in MPI message
     * \return request object, use this to wait for completion of the
     * non-blocking send
     */
    template<typename T>
    MPIRequest isend(const std::vector<T>& sbuf, int dest, int tag) const {
      MPIRequest req;
      // const_cast is necessary for ancient openmpi version used on Travis
      MPI_Isend(const_cast<T*>(sbuf.data()), sbuf.size(), mpi_type<T>(),
                dest, tag, comm_, req.req_.get());
      return req;
    }

    /**
     * Non-blocking send of a vector to a destination process, with a
     * certain tag.
     *
     * \tparam T template type of the send buffer, should have a
     * corresponding mpi_type<T>() implementation
     *
     * \param sbuf buffer of type T to be send
     * \param dest rank of destination process in this MPI communicator
     * \param tag tag to use in MPI message
     * \param req MPI request object
     */
    template<typename T>
    void isend(const std::vector<T>& sbuf, int dest, int tag,
               MPI_Request* req) const {
      // const_cast is necessary for ancient openmpi version used on Travis
      MPI_Isend(const_cast<T*>(sbuf.data()), sbuf.size(), mpi_type<T>(),
                dest, tag, comm_, req);
    }

    template<typename T>
    void isend(const T* sbuf, std::size_t ssize, int dest,
               int tag, MPI_Request* req) const {
      // const_cast is necessary for ancient openmpi version used on Travis
      MPI_Isend(const_cast<T*>(sbuf), ssize, mpi_type<T>(),
                dest, tag, comm_, req);
    }
    template<typename T>
    void send(const T* sbuf, std::size_t ssize, int dest, int tag) const {
      // const_cast is necessary for ancient openmpi version used on Travis
      MPI_Send(const_cast<T*>(sbuf), ssize, mpi_type<T>(), dest, tag, comm_);
    }

    template<typename T>
    void isend(const T& buf, int dest, int tag, MPI_Request* req) const {
      // const_cast is necessary for ancient openmpi version used on Travis
      MPI_Isend(const_cast<T*>(&buf), 1, mpi_type<T>(),
                dest, tag, comm_, req);
    }

    /**
     * Blocking send of a vector to a destination process, with a
     * certain tag.
     *
     * \tparam T template type of the send buffer, should have a
     * corresponding mpi_type<T>() implementation
     *
     * \param sbuf buffer of type T to be send
     * \param dest rank of destination process in this MPI communicator
     * \param tag tag to use in MPI message
     *
     * \see recv, isend
     */
    template<typename T>
    void send(const std::vector<T>& sbuf, int dest, int tag) const {
      // const_cast is necessary for ancient openmpi version used on Travis
      MPI_Send(const_cast<T*>(sbuf.data()), sbuf.size(),
               mpi_type<T>(), dest, tag, comm_);
    }

    /**
     * Receive a vector of T's from process src, with tag. The message
     * size does not need to be known in advance.
     *
     * \tparam T template parameter of vector to receive, should have
     * a corresponding mpi_type<T>() implementation
     *
     * \param src process to receive from
     * \param tag tag to match the message
     * \return std::vector<T> with the data to be received.
     * \see isend, send
     */
    template<typename T> std::vector<T> recv(int src, int tag) const {
      MPI_Status stat;
      MPI_Probe(src, tag, comm_, &stat);
      int msgsize;
      MPI_Get_count(&stat, mpi_type<T>(), &msgsize);
      //std::vector<T,NoInit<T>> rbuf(msgsize);
      std::vector<T> rbuf(msgsize);
      MPI_Recv(rbuf.data(), msgsize, mpi_type<T>(), src, tag,
               comm_, MPI_STATUS_IGNORE);
      return rbuf;
    }

    template<typename T>
    std::pair<int,std::vector<T>> recv_any_src(int tag) const {
      MPI_Status stat;
      MPI_Probe(MPI_ANY_SOURCE, tag, comm_, &stat);
      int msgsize;
      MPI_Get_count(&stat, mpi_type<T>(), &msgsize);
      std::vector<T> rbuf(msgsize);
      MPI_Recv(rbuf.data(), msgsize, mpi_type<T>(), stat.MPI_SOURCE,
               tag, comm_, MPI_STATUS_IGNORE);
      return {stat.MPI_SOURCE, std::move(rbuf)};
    }

    template<typename T> T recv_one(int src, int tag) const {
      T t;
      MPI_Recv(&t, 1, mpi_type<T>(), src, tag, comm_, MPI_STATUS_IGNORE);
      return t;
    }

    template<typename T>
    void irecv(const T* rbuf, std::size_t rsize, int src,
               int tag, MPI_Request* req) const {
      // const_cast is necessary for ancient openmpi version used on Travis
      MPI_Irecv(const_cast<T*>(rbuf), rsize, mpi_type<T>(),
                src, tag, comm_, req);
    }

    template<typename T>
    void recv(const T* rbuf, std::size_t rsize, int src, int tag) const {
      // const_cast is necessary for ancient openmpi version used on Travis
      MPI_Status stat;
      MPI_Recv(const_cast<T*>(rbuf), rsize, mpi_type<T>(),
               src, tag, comm_, &stat);
    }

    /**
     * Compute the reduction of op(t_i) over all processes i, where op
     * can be any MPI_Op, on all processes. This routine is collective
     * on all ranks in this communicator. See documentation for
     * MPI_Allreduce.
     *
     * \tparam T type of variable to reduce, should have a
     * corresponding mpi_type<T>() implementation
     *
     * \param t variable to reduce, passed by value
     * \param op reduction operator
     * \return variable of type T, result of the reduction, available
     * on each rank
     */
    template<typename T> T all_reduce(T t, MPI_Op op) const {
      MPI_Allreduce(MPI_IN_PLACE, &t, 1, mpi_type<T>(), op, comm_);
      return t;
    }

    /**
     * Compute the reduction of op(t_i) over all processes i, where op
     * can be any MPI_Op, on the root. This routine is collective
     * on all ranks in this communicator. See documentation for
     * MPI_Reduce (with dest == 0).
     *
     * \tparam T type of variable to reduce, should have a
     * corresponding mpi_type<T>() implementation
     *
     * \param t variable to reduce, passed by value
     * \param op reduction operator
     * \return variable of type T, result of the reduction, available
     * only on the root process.
     */
    template<typename T> T reduce(T t, MPI_Op op) const {
      MPI_Reduce(is_root() ? MPI_IN_PLACE : &t, &t, 1,
                 mpi_type<T>(), op, 0, comm_);
      return t;
    }

    /**
     * Compute the reduction of op(t[]_i) over all processes i, t[] is
     * an array, and where op can be any MPI_Op, on all
     * processes. This routine is collective on all ranks in this
     * communicator. See documentation for MPI_Allreduce. Every
     * element of the array will be reduced over the different
     * processes. The operation is performed in-place.
     *
     * \tparam T type of variables to reduce, should have a
     * corresponding mpi_type<T>() implementation
     *
     * \param t pointer to array of variables to reduce
     * \param ssize size of array to reduce
     * \param op reduction operator
     */
    template<typename T> void all_reduce(T* t, int ssize, MPI_Op op) const {
      MPI_Allreduce(MPI_IN_PLACE, t, ssize, mpi_type<T>(), op, comm_);
    }

    template<typename T> void all_reduce(std::vector<T>& t, MPI_Op op) const {
      all_reduce(t.data(), t.size(), op);
    }

    /**
     * Compute the reduction of op(t[]_i) over all processes i, t[] is
     * an array, and where op can be any MPI_Op, on the root
     * process. This routine is collective on all ranks in this
     * communicator. See documentation for MPI_Reduce. Every element
     * of the array will be reduced over the different processes. The
     * operation is performed in-place.
     *
     * \tparam T type of variables to reduce, should have a
     * corresponding mpi_type<T>() implementation
     *
     * \param t pointer to array of variables to reduce
     * \param ssize size of array to reduce
     * \param op reduction operator
     * \param dest where to reduce to
     */
    template<typename T> void
    reduce(T* t, int ssize, MPI_Op op, int dest=0) const {
      MPI_Reduce(rank() == dest ? MPI_IN_PLACE : t, t, ssize,
                 mpi_type<T>(), op, dest, comm_);
    }

    template<typename T>
    void all_to_all(const T* sbuf, int scnt, T* rbuf) const {
      MPI_Alltoall
        (sbuf, scnt, mpi_type<T>(), rbuf, scnt, mpi_type<T>(), comm_);
    }

    template<typename T, typename A=std::allocator<T>> std::vector<T,A>
    all_to_allv(const T* sbuf, int* scnts, int* sdispls,
                int* rcnts, int* rdispls) const {
      std::size_t rsize = 0;
      for (int p=0; p<size(); p++)
        rsize += rcnts[p];
      std::vector<T,A> rbuf(rsize);
      MPI_Alltoallv
        (sbuf, scnts, sdispls, mpi_type<T>(),
         rbuf.data(), rcnts, rdispls, mpi_type<T>(), comm_);
      return rbuf;
    }

    template<typename T> void
    all_to_allv(const T* sbuf, int* scnts, int* sdispls,
                T* rbuf, int* rcnts, int* rdispls) const {
      MPI_Alltoallv
        (sbuf, scnts, sdispls, mpi_type<T>(),
         rbuf, rcnts, rdispls, mpi_type<T>(), comm_);
    }

    /**
     * Perform an MPI_Alltoallv. Each rank sends sbuf[i] to process
     * i. The results are received in a single contiguous vector
     * rbuf. pbuf has pointers into rbuf, with pbuf[i] pointing to the
     * data received from rank i.  This is collective on this MPI
     * communicator.
     *
     * \tparam T type of data to send, this should have a
     * corresponding mpi_type<T>() implementation or should define
     * T::mpi_type()
     * \tparam A allocator to be used for the rbuf
     * \param sbuf send buffers (should be size this->size())
     * \param rbuf receive buffer, can be empty, will be allocated
     * \param pbuf pointers (to positions in rbuf) to where data
     * received from different ranks start
     * \see all_to_all_v
     */
    template<typename T, typename A=std::allocator<T>> void
    all_to_all_v(std::vector<std::vector<T>>& sbuf, std::vector<T,A>& rbuf,
                 std::vector<T*>& pbuf) const {
      all_to_all_v(sbuf, rbuf, pbuf, mpi_type<T>());
    }

    /**
     * Perform an MPI_Alltoallv. Each rank sends sbuf[i] to process
     * i. The results are received in a single contiguous vector which
     * is returned. This is collective on this MPI communicator.
     *
     * \tparam T type of data to send, this should have a
     * corresponding mpi_type<T>() implementation or should define
     * T::mpi_type()
     * \tparam A allocator to be used for the receive buffer
     * \param sbuf send buffers (should be size this->size())
     * \return receive buffer
     * \see all_to_all_v
     */
    template<typename T, typename A=std::allocator<T>> std::vector<T,A>
    all_to_all_v(std::vector<std::vector<T>>& sbuf) const {
      std::vector<T,A> rbuf;
      std::vector<T*> pbuf;
      all_to_all_v(sbuf, rbuf, pbuf, mpi_type<T>());
      return rbuf;
    }

    /**
     * Perform an MPI_Alltoallv. Each rank sends sbuf[i] to process
     * i. The results are received in a single contiguous vector
     * rbuf. pbuf has pointers into rbuf, with pbuf[i] pointing to the
     * data received from rank i.  This is collective on this MPI
     * communicator.
     *
     * \tparam T type of data to send
     * \tparam A allocator to be used for the receive buffer
     * \param sbuf send buffers (should be size this->size())
     * \param rbuf receive buffer, can be empty, will be allocated
     * \param pbuf pointers (to positions in rbuf) to where data
     * received from different ranks start
     * \param Ttype MPI_Datatype corresponding to the template
     * parameter T
     * \see all_to_all_v
     */
    template<typename T, typename A=std::allocator<T>> void
    all_to_all_v(std::vector<std::vector<T>>& sbuf, std::vector<T,A>& rbuf,
                 std::vector<T*>& pbuf, const MPI_Datatype Ttype) const {
      assert(sbuf.size() == std::size_t(size()));
      auto P = size();
      std::unique_ptr<int[]> iwork(new int[4*P]);
      auto ssizes = iwork.get();
      auto rsizes = ssizes + P;
      auto sdispl = ssizes + 2*P;
      auto rdispl = ssizes + 3*P;
      for (int p=0; p<P; p++) {
        if (sbuf[p].size() >
            static_cast<std::size_t>(std::numeric_limits<int>::max())) {
          std::cerr << "# ERROR: 32bit integer overflow in all_to_all_v!!"
                    << std::endl;
          MPI_Abort(comm_, 1);
        }
        ssizes[p] = sbuf[p].size();
      }
      MPI_Alltoall
        (ssizes, 1, mpi_type<int>(), rsizes, 1, mpi_type<int>(), comm_);
      std::size_t totssize = std::accumulate(ssizes, ssizes+P, std::size_t(0)),
        totrsize = std::accumulate(rsizes, rsizes+P, std::size_t(0));
#if 1
      if (true) {
        // Always implement the all_to_all_v with Irecv/Isend loops.
        // We noticed (on NERSC Cori) that MPI_Alltoallv gave wrong
        // results for large problems (using double complex), likely
        // due to some overflow, even tho the totssize/totrsize was <
        // MAX_INT. Also, using Irecv/Isend directly avoids to copies
        // from the separate send buffers to the single send buffer.
#else
      if (totrsize >
        static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        totssize >
        static_cast<std::size_t>(std::numeric_limits<int>::max())) {
#endif
        // This case will probably cause an overflow in the
        // rdispl/sdispl elements. Here we do the all_to_all_v
        // manually by just using Isend/Irecv. This might be slower
        // than splitting into multiple calls to MPI_Alltoallv
        // (although it avoids a copy from the sbuf).
        rbuf.resize(totrsize);
        std::unique_ptr<MPI_Request[]> reqs(new MPI_Request[2*P]);
        std::size_t displ = 0;
        pbuf.resize(P);
        int r = rank();
        for (int p=0; p<P; p++) {
          auto dst = (r + p) % P;
          pbuf[dst] = rbuf.data() + displ;
          MPI_Irecv(pbuf[dst], rsizes[dst], Ttype, dst, 0, comm_, reqs.get()+dst);
          displ += rsizes[dst];
        }
        for (int p=0; p<P; p++) {
          auto dst = (r + p) % P;
          MPI_Isend
            (sbuf[dst].data(), ssizes[dst], Ttype, dst, 0, comm_, reqs.get()+P+dst);
        }
        MPI_Waitall(2*P, reqs.get(), MPI_STATUSES_IGNORE);
        std::vector<std::vector<T>>().swap(sbuf);
      } else {
        std::unique_ptr<T[]> sendbuf_(new T[totssize]);
        auto sendbuf = sendbuf_.get();
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
      }
    }

    /**
     * Return a subcommunicator with P ranks, starting from rank P0,
     * using stride stride. Ie., ranks (relative to this communicator)
     * [P0:stride:P0+stride*P) will be included in the new
     * communicator.  This operation is collective on all the
     * processes in this communicator.
     *
     * \param P0 first rank in the new communicator
     * \param P number of ranks in the new communicator
     * \param stride stride between ranks in this communicator
     * determining which ranks will go into new communicator
     * \return new communicator containing P ranks from this
     * communicator [P0:stride:P0+stride*P)
     * \see sub_self
     */
    MPIComm sub(int P0, int P, int stride=1) const {
      if (is_null()) return MPIComm(MPI_COMM_NULL);
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

    /**
     * Returns a communicator with only rank p, or an MPIComm wrapping
     * MPI_COMM_NULL if my rank in the current MPIComm != p. This is
     * collective on the current communicator.
     *
     * \param p rank to be included in new MPIComm
     * \return a new communicator containing only rank p (from the
     * original MPIComm)
     */
    // return MPI_COMM_SELF??? or MPI_COMM_NULL if not rank??
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

    /**
     * Call MPI_Pcontrol with level 1, and string name
     */
    static void control_start(const std::string& name) {
      MPI_Pcontrol(1, name.c_str());
    }
    /**
     * Call MPI_Pcontrol with level -1, and string name
     */
    static void control_stop(const std::string& name) {
      MPI_Pcontrol(-1, name.c_str());
    }

    static bool initialized() {
      int flag;
      MPI_Initialized(&flag);
      return static_cast<bool>(flag);
    }

  private:
    MPI_Comm comm_ = MPI_COMM_WORLD;

    void duplicate(MPI_Comm c) {
      if (c == MPI_COMM_NULL) comm_ = c;
      else MPI_Comm_dup(c, &comm_);
    }
  };


  /**
   * Return this process rank in MPI communicator c, or in
   * MPI_COMM_WORLD if c is not provided.
   *
   * This routine is deprecated, will be removed soon. USe the MPIComm
   * interface instead.
   */
  inline int mpi_rank(MPI_Comm c=MPI_COMM_WORLD) {
    assert(c != MPI_COMM_NULL);
    int rank;
    MPI_Comm_rank(c, &rank);
    return rank;
  }

  /**
   * Return the number of processes in MPI communicator c, or in
   * MPI_COMM_WORLD if c is not provided.
   *
   * This routine is deprecated, will be removed soon. USe the MPIComm
   * interface instead.
   */
  inline int mpi_nprocs(MPI_Comm c=MPI_COMM_WORLD) {
    assert(c != MPI_COMM_NULL);
    int nprocs;
    MPI_Comm_size(c, &nprocs);
    return nprocs;
  }

} // end namespace strumpack

#endif // STRUMPACK_MPI_WRAPPER_HPP
