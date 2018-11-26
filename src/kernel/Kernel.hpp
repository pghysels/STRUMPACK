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
 * \file Kernel.hpp
 *
 * \brief Definitions of several kernel functions, and helper
 * routines. Also provides driver routines for kernel ridge
 * regression.
 */
#ifndef STRUMPACK_KERNEL_HPP
#define STRUMPACK_KERNEL_HPP

#include "Metrics.hpp"
#include "HSS/HSSOptions.hpp"
#include "dense/DenseMatrix.hpp"
#if defined(STRUMPACK_USE_MPI)
#include "dense/DistributedMatrix.hpp"
#if defined(STRUMPACK_USE_HODLRBF)
#include "HODLR/HODLROptions.hpp"
#endif
#endif

namespace strumpack {

  /**
   * Defines simple kernel matrix definitions and kernel regression.
   */
  namespace kernel {

    /**
     * Representation of a kernel matrix. This is an abstract class.
     * This class contains a reference to the datapoints representing
     * the kernel: X. X is a d x n matrix (d features and n
     * datapoints).
     *
     */
    template<typename scalar_t> class Kernel {
      using DenseM_t = DenseMatrix<scalar_t>;
      using DenseMW_t = DenseMatrixWrapper<scalar_t>;
#if defined(STRUMPACK_USE_MPI)
      using DistM_t = DistributedMatrix<scalar_t>;
#endif

    public:
      /**
       * Kernel matrix constructor.
       *
       * \param data Contains the data points, 1 datapoint per column.
       * So the number of rows of data is the number of features, the
       * number of columns of data will be the size of the kernel
       * matrix.
       * \param lambda regularization parameter, added to the
       * diagonal.
       */
      Kernel(DenseM_t& data, scalar_t lambda)
        : data_(data), lambda_(lambda) { }

      /**
       * Default constructor.
       */
      virtual ~Kernel() = default;

      /**
       * Returns the size of the (square) kernel matrix. This is the
       * number of datapoints in the set data().
       *
       * \see d(), data()
       */
      std::size_t n() const { return data_.cols(); }

      /**
       * Return the dimension of the datapoints defining the kernel.
       * \return dimension of the datapoints
       * \see n(), data()
       */
      std::size_t d() const { return data_.rows(); }

      /**
       * Evaluate an entry of the kernel matrix.
       *
       * \param m row coordinate of entry to evaluate
       * \param n column coordinate of entry to evaluate
       * \return the value K(m, n) of the kernel
       */
      scalar_t eval(std::size_t i, std::size_t j) const {
        return eval_kernel_function(data_.ptr(0, i), data_.ptr(0, j))
          + ((i == j) ? lambda_ : scalar_t(0.));
      }

      /**
       * Evaluate multiple entries at once: evaluate the submatrix
       * K(I,J) and put the result in matrix B.
       *
       * \param I set of row indices of elements to extract
       * \param J set of col indices of elements to extract
       * \return K(I,J), matrix B should be the correct size, ie.,
       * B.rows() == I.size() and B.cols() == J.size()
       */
      void operator()(const std::vector<std::size_t>& I,
                      const std::vector<std::size_t>& J,
                      DenseM_t& B) const {
        assert(B.rows() == I.size() && B.cols() == J.size());
        for (auto j=0; j<J.size(); j++)
          for (auto i=0; i<I.size(); i++) {
            assert(I[i] < n() && J[j] < n());
            B(i, j) = eval(I[i], J[j]);
          }
      }

      /**
       * TODO describe
       * data will get permuted, together with the labels
       */
      DenseM_t fit_HSS
      (std::vector<scalar_t>& labels, const HSS::HSSOptions<scalar_t>& opts);

      /**
       * TODO describe
       * weights is a column vector (can be multiple vectors)
       */
      std::vector<scalar_t> predict
      (const DenseM_t& test, const DenseM_t& weights) const;

#if defined(STRUMPACK_USE_MPI)
      /**
       * TODO describe
       * data will get permuted, together with the labels
       */
      DistM_t fit_HSS
      (const BLACSGrid& grid, std::vector<scalar_t>& labels,
       const HSS::HSSOptions<scalar_t>& opts);

      /**
       * TODO describe
       * weights is a column vector (can be multiple vectors)
       */
      std::vector<scalar_t> predict
      (const DenseM_t& test, const DistM_t& weights) const;

#if defined(STRUMPACK_USE_HODLRBF)

      /**
       * What does this return? The local part of the 1D block row
       * distributed weights? Or the whole weights vector on every
       * rank?
       */
      DenseM_t fit_HODLR
      (const MPIComm& c, std::vector<scalar_t>& labels,
       const HODLR::HODLROptions<scalar_t>& opts);

#endif
#endif

      /**
       * Returns a (const) reference to the data used to define this
       * kernel.
       * \return const reference to the datapoint, a matrix of size d
       * x n.
       */
      const DenseM_t& data() const { return data_; }
      /**
       * Returns a reference to the data used to define this
       * kernel.
       * \return reference to the datapoint, a matrix of size d x n.
       */
      DenseM_t& data() { return data_; }

    protected:
      DenseM_t& data_;
      scalar_t lambda_;

      virtual scalar_t eval_kernel_function
      (const scalar_t* x, const scalar_t* y) const = 0;
    };


    /**
     * Gaussian (radial basis function) kernel.
     *
     * \see Kernel
     */
    template<typename scalar_t>
    class GaussKernel : public Kernel<scalar_t> {
    public:
      GaussKernel(DenseMatrix<scalar_t>& data, scalar_t h, scalar_t lambda)
        : Kernel<scalar_t>(data, lambda), h_(h) {}

    protected:
      scalar_t h_; // kernel width parameter

      scalar_t eval_kernel_function
      (const scalar_t* x, const scalar_t* y) const override {
        return std::exp
          (-Euclidean_distance_squared(this->d(), x, y)
           / (scalar_t(2.) * h_ * h_));
      }
    };


    /**
     * Laplace kernel.
     *
     * \see GaussKernel, Kernel
     */
    template<typename scalar_t>
    class LaplaceKernel : public Kernel<scalar_t> {
    public:
      LaplaceKernel(DenseMatrix<scalar_t>& data, scalar_t h, scalar_t lambda)
        : Kernel<scalar_t>(data, lambda), h_(h) {}

    protected:
      scalar_t h_; // kernel width parameter

      scalar_t eval_kernel_function
      (const scalar_t* x, const scalar_t* y) const override {
        return std::exp(-norm1_distance(this->d(), x, y) / h_);
      }
    };



    /**
     * Enumeration of Kernel types.
     * \ingroup Enumerations
     */
    enum class KernelType {
      GAUSS,   /*!< Gauss or radial basis function kernel */
      LAPLACE  /*!< Laplace kernel                        */
    };

    /**
     * Return a string with the name of the kernel type.
     */
    inline std::string get_name(KernelType k) {
      switch (k) {
      case KernelType::GAUSS: return "Gauss"; break;
      case KernelType::LAPLACE: return "Laplace"; break;
      default: return "UNKNOWN";
      }
    }

    /**
     * Return a KernelType enum from a string. If the string is not
     * recognized, a warning is printed and the GAUSS kernel type is
     * returned.
     */
    inline KernelType kernel_type(const std::string& k) {
      if (k.compare("Gauss") == 0)
        return KernelType::GAUSS;
      else if (k.compare("Laplace"))
        return KernelType::LAPLACE;
      std::cerr << "ERROR: Kernel type not recogonized, "
                << " setting kernel type to GAUSS."
                << std::endl;
      return KernelType::GAUSS;
    }

    /**
     * Creates a unique_ptr to a Kernel object, which can be
     * GaussKernel, LaplaceKernel, ... .
     *
     * \tparam scalar_t the scalar type to represent the kernel.
     *
     * \param args arguments to be passed through to the constructor
     * of the actual kernel (fi, data, h, lambda for GaussKernel)
     *
     * \return unique_ptr to a kernel
     */
    template<typename scalar_t, typename ... Args>
    std::unique_ptr<Kernel<scalar_t>> create_kernel
    (KernelType k, Args& ... args) {
      switch (k) {
      case KernelType::GAUSS:
        return std::unique_ptr<Kernel<scalar_t>>
          (new GaussKernel<scalar_t>(args ...));
      case KernelType::LAPLACE:
        return std::unique_ptr<Kernel<scalar_t>>
          (new LaplaceKernel<scalar_t>(args ...));
      default:
        return std::unique_ptr<Kernel<scalar_t>>
          (new GaussKernel<scalar_t>(args ...));
      }
    }

  } // end namespace kernel

} // end namespace strumpack

#endif // STRUMPACK_KERNEL_HPP
