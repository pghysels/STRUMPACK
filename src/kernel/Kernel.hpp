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
#if defined(STRUMPACK_USE_BPACK)
#include "HODLR/HODLROptions.hpp"
#endif
#endif

namespace strumpack {

  /**
   * Defines simple kernel matrix definitions and kernel regression.
   */
  namespace kernel {

    /**
     * \class Kernel
     *
     * \brief Representation of a kernel matrix.
     *
     * This is an abstract class. This class contains a reference to
     * the datapoints representing the kernel: X. X is a d x n matrix
     * (d features and n datapoints).
     *
     * The actual kernel function is implemented in one of the
     * subclasses of this class. Subclasses need to implement the
     * purely virtual function eval_kernel_function.
     *
     * \tparam scalar_t Scalar type of the input data points and the
     * kernel representation. Can be float, double,
     * std::complex<float> or std::complex<double>.
     */
    template<typename scalar_t> class Kernel {
      using real_t = typename RealType<scalar_t>::value_type;
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
       * \param i row coordinate of entry to evaluate
       * \param j column coordinate of entry to evaluate
       * \return the value K(i, j) of the kernel
       */
      virtual scalar_t eval(std::size_t i, std::size_t j) const {
        return eval_kernel_function(data_.ptr(0, i), data_.ptr(0, j))
          + ((i == j) ? lambda_ : scalar_t(0.));
      }

      /**
       * Evaluate multiple entries at once: evaluate the submatrix
       * K(I,J) and put the result in matrix B. This is used in the
       * HSS and HODLR construction algorithms for this kernel.
       *
       * \param I set of row indices of elements to extract
       * \param J set of col indices of elements to extract
       * \param B B will be set to K(I,J). Matrix B should be the
       * correct size, ie., B.rows() == I.size() and B.cols() ==
       * J.size()
       */
      void operator()(const std::vector<std::size_t>& I,
                      const std::vector<std::size_t>& J,
                      DenseMatrix<real_t>& B) const {
        assert(B.rows() == I.size() && B.cols() == J.size());
        for (std::size_t j=0; j<J.size(); j++)
          for (std::size_t i=0; i<I.size(); i++) {
            assert(I[i] < n() && J[j] < n());
            B(i, j) = eval(I[i], J[j]);
          }
      }

      /**
       * Evaluate multiple entries at once: evaluate the submatrix
       * K(I,J) and put the result in matrix B. This is used in the
       * HSS and HODLR construction algorithms for this kernel.
       *
       * \param I set of row indices of elements to extract
       * \param J set of col indices of elements to extract
       * \param B B will be set to K(I,J). Matrix B should be the
       * correct size, ie., B.rows() == I.size() and B.cols() ==
       * J.size()
       */
      void operator()(const std::vector<std::size_t>& I,
                      const std::vector<std::size_t>& J,
                      DenseMatrix<std::complex<real_t>>& B) const {
        assert(B.rows() == I.size() && B.cols() == J.size());
        for (std::size_t j=0; j<J.size(); j++)
          for (std::size_t i=0; i<I.size(); i++) {
            assert(I[i] < n() && J[j] < n());
            B(i, j) = eval(I[i], J[j]);
          }
      }

      /**
       * Compute weights for kernel ridge regression
       * classification. This will build an approximate HSS
       * representation of the kernel matrix and use that to solve a
       * linear system with the kernel matrix and the weights vector
       * as the right-hand side. The weights can then later be used in
       * the predict method for prediction. The data associated to
       * this kernel, and the labels, will get permuted.
       *
       * __TODO__ labels should be a vector of bool's or int's??
       *
       * \param labels Binary labels, supposed to be in {-1, 1}.
       * Should be labels.size() == this->n().
       * \param opts HSS options
       * \return A vector (1 column matrix) with scalar weights, to be
       * used in predict
       * \see predict, fit_HODLR
       */
      DenseM_t fit_HSS
      (std::vector<scalar_t>& labels, const HSS::HSSOptions<scalar_t>& opts);

      /**
       * Return prediction scores for the test points, using the
       * weights computed in fit_HSS() or fit_HODLR().
       *
       * \param test Test data set, should be test.rows() == this->d()
       * \param weights Weights computed by fit_HSS() or fit_HODLR()
       * \return Vector with prediction scores. One can use the sign
       * (threshold zero), to decide which of 2 classes each test
       * point belongs to.
       * \see fit_HSS, fit_HODLR
       */
      std::vector<scalar_t> predict
      (const DenseM_t& test, const DenseM_t& weights) const;

#if defined(STRUMPACK_USE_MPI)
      /**
       * Compute weights for kernel ridge regression
       * classification. This will build an approximate HSS
       * representation of the kernel matrix and use that to solve a
       * linear system with the kernel matrix and the weights vector
       * as the right-hand side. The weights can then later be used in
       * the predict() method for prediction. The data associated to
       * this kernel, and the labels, will get permuted.
       *
       * __TODO__ labels should be a vector of bool's or int's??
       *
       * \param grid Processor grid to use for the MPI distributed
       * computations
       * \param labels Binary labels, supposed to be in {-1, 1}.
       * Should be labels.size() == this->n().
       * \param opts HSS options
       * \return A distributed vector with scalar weights, to be used
       * in predict(), distributed on the BLACSGrid grid.
       * \see predict, fit_HODLR
       */
      DistM_t fit_HSS
      (const BLACSGrid& grid, std::vector<scalar_t>& labels,
       const HSS::HSSOptions<scalar_t>& opts);

      /**
       * Return prediction scores for the test points, using the
       * weights computed in fit_HSS() or fit_HODLR().
       *
       * \param test Test data set, should be test.rows() == this->d()
       * \param weights Weights computed by fit_HSS() or fit_HODLR()
       * \return Vector with prediction scores. One can use the sign
       * (threshold zero), to decide which of 2 classes each test
       * point belongs to.
       * \see fit_HSS, fit_HODLR
       */
      std::vector<scalar_t> predict
      (const DenseM_t& test, const DistM_t& weights) const;

#if defined(STRUMPACK_USE_BPACK)
      /**
       * Compute weights for kernel ridge regression
       * classification. This will build an approximate HODLR
       * representation of the kernel matrix and use that to solve a
       * linear system with the kernel matrix and the weights vector
       * as the right-hand side. The weights can then later be used in
       * the predict() method for prediction. The data associated to
       * this kernel, and the labels, will get permuted.
       *
       * __TODO__ labels should be a vector of bool's or int's??
       *
       * \param c MPI communicator on which to perform the calculation
       * \param labels Binary labels, supposed to be in {-1, 1}.
       * Should be labels.size() == this->n().
       * \param opts HSS options
       * \return A vector with scalar weights, to be used in predict()
       * \see predict, fit_HSS
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

      std::vector<int>& permutation() { return perm_; }
      const std::vector<int>& permutation() const { return perm_; }

      virtual void permute() {}

    protected:
      DenseM_t& data_;
      scalar_t lambda_;
      std::vector<int> perm_;

      /**
       * Purely virtual function that needs to be defined in the
       * subclass. This defines the actual kernel function. All data
       * points will be of dimension d(), and one can use several
       * already defined distances, such as
       * Euclidean_distance_squared, Euclidean_distance, or
       * norm1_distance.
       *
       * \param x Pointer to first data point, of dimension d().
       * \param y Pointer to second data point, of dimension d().
       * \return Evaluation of kernel function
       * \see GaussKernel::eval_kernel_function,
       * LaplacKernel::eval_kernel_function
       */
      virtual scalar_t eval_kernel_function
      (const scalar_t* x, const scalar_t* y) const = 0;
    };


    /**
     * \class GaussKernel
     *
     * \brief Gaussian or radial basis function kernel.
     *
     * Implements the kernel: \f$\exp \left( -\frac{\|x-y\|_2^2}{2
     * h^2} \right)\f$, with an extra regularization parameter lambda
     * on the diagonal.
     *
     * This is a subclass of Kernel. It only implements the
     * (protected) eval_kernel_function routine, the rest of the
     * functionality is inherited. To create your own kernel, simply
     * copy this class, rename and change the eval_kernel_function
     * implementation.
     *
     * \see Kernel, LaplaceKernel
     */
    template<typename scalar_t>
    class GaussKernel : public Kernel<scalar_t> {
    public:
      /**
       * Constructor of the kernel object.
       *
       * \param data Data defining the kernel matrix. data.rows() is
       * the number of features, and data.cols() is the number of data
       * points, ie, the dimension of the kernel matrix.
       *
       * \param h Kernel width
       * \param lambda Regularization parameter, added to the diagonal
       */
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
     * \class LaplaceKernel
     *
     * \brief Laplace kernel.
     *
     * Implements the kernel: \f$\exp \left( -\frac{\|x-y\|_1}{h}
     * \right)\f$, with an extra regularization parameter lambda on
     * the diagonal.
     *
     * This is a subclass of Kernel. It only implements the
     * (protected) eval_kernel_function routine, the rest of the
     * functionality is inherited. To create your own kernel, simply
     * copy this class, rename and change the eval_kernel_function
     * implementation.
     *
     * \see Kernel, GaussKernel
     */
    template<typename scalar_t>
    class LaplaceKernel : public Kernel<scalar_t> {
    public:
      /**
       * Constructor of the kernel object.
       *
       * \param data Data defining the kernel matrix. data.rows() is
       * the number of features, and data.cols() is the number of data
       * points, ie, the dimension of the kernel matrix.
       *
       * \param h Kernel width
       * \param lambda Regularization parameter, added to the diagonal
       */
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
     * \class ANOVAKernel
     *
     * \brief ANOVA kernel.
     *
     * Implements the kernel: \f$ \sum_{1\leq k_1<...<k_p\leq
     * n}k(x^{k_1},y^{k_1})\times...\times k(x^{k_p},y^{k_p}), with
     * k(x,y)=\exp \left( -\frac{\|x-y\|_2^2}{2 h^2} \right)\f$, with
     * an extra regularization parameter lambda on the diagonal. The
     * kernel is implemented efficiently due to the recurrence
     * relation in "Support Vector Regression with ANOVA Decomposition
     * Kernels", 1999.
     *
     * This is a subclass of Kernel. It only implements the
     * (protected) eval_kernel_function routine, the rest of the
     * functionality is inherited. To create your own kernel, simply
     * copy this class, rename and change the eval_kernel_function
     * implementation.
     *
     * \see Kernel, GaussKernel
     */
    template<typename scalar_t>
    class ANOVAKernel : public Kernel<scalar_t> {
    public:
      /**
       * Constructor of the kernel object.
       *
       * \param data Data defining the kernel matrix. data.rows() is
       * the number of features, and data.cols() is the number of data
       * points, ie, the dimension of the kernel matrix.
       *
       * \param h Kernel width
       * \param lambda Regularization parameter, added to the diagonal
       * \param p Kernel degree (1 <= p_ <= this->d())
       */
      ANOVAKernel
      (DenseMatrix<scalar_t>& data, scalar_t h, scalar_t lambda, int p=1)
        : Kernel<scalar_t>(data, lambda), h_(h), p_(p) {
        assert(p >= 1 && p <= int(this->d()));
      }

    protected:
      scalar_t h_; // kernel width parameter
      int p_;      // kernel degree parameter 1 <= p_ <= this->d()

      scalar_t eval_kernel_function
      (const scalar_t* x, const scalar_t* y) const override {
        std::vector<scalar_t> Ks(p_), Kss(p_), Kpp(p_+1);
        Kpp[0] = 1;
        for (int j=0; j<p_; j++) Kss[j] = 0;
        for (std::size_t i=0; i<this->d(); i++) {
          scalar_t tmp = std::exp
            (-Euclidean_distance_squared(1, x+i, y+i)
             / (scalar_t(2.) * h_ * h_));
          Ks[0] = tmp;
          Kss[0] += Ks[0];
          for (int j=1; j<p_; j++) {
            Ks[j] = Ks[j-1]*tmp;
            Kss[j] += Ks[j];
          }
        }
        for (int i=1; i<=p_; i++) {
          Kpp[i] = 0;
          for (int s=1; s<=i; s++)
            Kpp[i] += std::pow(-1,s+1)*Kpp[i-s]*Kss[s-1];
          Kpp[i] /= i;
        }
        return Kpp[p_];
      }
    };


    /**
     * \class DenseKernel
     *
     * \brief Arbitrary dense matrix, with underlying geometry.
     *
     * This is a subclass of Kernel. It overrides the eval routine,
     * unlike other kernel classes (Gauss and Laplace), which only
     * implement the kernel function.
     *
     * \see Kernel, GaussKernel, LaplaceKernel
     */
    template<typename scalar_t>
    class DenseKernel : public Kernel<scalar_t> {
    public:
      /**
       * Constructor of the dense matrix kernel object.
       *
       * \param data Coordinates of the points used to generate the
       * matrix.
       *
       * \param A The actual dense matrix
       *
       * \param lambda Regularization parameter, added to the diagonal
       */
      DenseKernel(DenseMatrix<scalar_t>& data,
                  DenseMatrix<scalar_t>& A, scalar_t lambda)
        : Kernel<scalar_t>(data, lambda), A_(A) {}

      scalar_t eval(std::size_t i, std::size_t j) const override {
        return A_(i, j) + ((i == j) ? this->lambda_ : scalar_t(0.));
      }

      void permute() override {
        A_.lapmt(this->perm_, true);
        A_.lapmr(this->perm_, true);
      }

    protected:
      DenseMatrix<scalar_t>& A_; // kernel matrix

      scalar_t eval_kernel_function
      (const scalar_t* x, const scalar_t* y) const override {
        assert(false);
      }
    };


    /**
     * Enumeration of Kernel types.
     * \ingroup Enumerations
     */
    enum class KernelType {
      DENSE,    /*!< Arbitrary dense matrix                */
      GAUSS,    /*!< Gauss or radial basis function kernel */
      LAPLACE,  /*!< Laplace kernel                        */
      ANOVA     /*!< ANOVA kernel                          */
    };

    /**
     * Return a string with the name of the kernel type.
     */
    inline std::string get_name(KernelType k) {
      switch (k) {
      case KernelType::DENSE: return "dense";
      case KernelType::GAUSS: return "Gauss";
      case KernelType::LAPLACE: return "Laplace";
      case KernelType::ANOVA: return "ANOVA";
      default: return "UNKNOWN";
      }
    }

    /**
     * Return a KernelType enum from a string. If the string is not
     * recognized, a warning is printed and the GAUSS kernel type is
     * returned.
     */
    inline KernelType kernel_type(const std::string& k) {
      if (k == "dense") return KernelType::DENSE;
      else if (k == "Gauss") return KernelType::GAUSS;
      else if (k == "Laplace") return KernelType::LAPLACE;
      else if (k == "ANOVA") return KernelType::ANOVA;
      std::cerr << "ERROR: Kernel type not recogonized, "
                << " setting kernel type to Gauss."
                << std::endl;
      return KernelType::GAUSS;
    }

    /**
     * Creates a unique_ptr to a Kernel object, which can be
     * GaussKernel, LaplaceKernel, ... .
     *
     * \tparam scalar_t the scalar type to represent the kernel.
     *
     * \param k Type of kernel
     * \param args arguments to be passed through to the constructor
     * of the actual kernel (fi, data, h, lambda for GaussKernel)
     *
     * \return unique_ptr to a kernel
     */
    template<typename scalar_t>
    std::unique_ptr<Kernel<scalar_t>> create_kernel
    (KernelType k, DenseMatrix<scalar_t>& data,
     scalar_t h, scalar_t lambda, int p=1) {
      switch (k) {
      // case KernelType::DENSE:
      //   return std::unique_ptr<Kernel<scalar_t>>
      //     (new DenseKernel<scalar_t>(args ...));
      case KernelType::GAUSS:
        return std::unique_ptr<Kernel<scalar_t>>
          (new GaussKernel<scalar_t>(data, h, lambda));
      case KernelType::LAPLACE:
        return std::unique_ptr<Kernel<scalar_t>>
          (new LaplaceKernel<scalar_t>(data, h, lambda));
      case KernelType::ANOVA:
        return std::unique_ptr<Kernel<scalar_t>>
          (new ANOVAKernel<scalar_t>(data, h, lambda, p));
      default:
        return std::unique_ptr<Kernel<scalar_t>>
          (new GaussKernel<scalar_t>(data, h, lambda));
      }
    }

  } // end namespace kernel

} // end namespace strumpack

#endif // STRUMPACK_KERNEL_HPP
