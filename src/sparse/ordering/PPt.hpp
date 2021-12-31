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
#ifndef STRUMPACK_ORDERING_PPT_HPP
#define STRUMPACK_ORDERING_PPT_HPP

#include <random>

#include "misc/Tools.hpp"
//#include "BLASLAPACKWrapper.hpp"

namespace strumpack {
  namespace ordering {

    template<typename intt> class PPt {
    public:
      PPt() = default;
      PPt(intt n) : n_(n), perm_(n), iperm_(n) {}

      intt n() const { return n_; }

      const intt* P() const { return perm_.data(); }
      intt* P() { return perm_.data(); }

      const intt* Pt() const { return iperm_.data(); }
      intt* Pt() { return iperm_.data(); }

      const intt& P(std::size_t i) const
      { assert(i<std::size_t(n())); return perm_[i]; }
      intt& P(std::size_t i)
      { assert(i<std::size_t(n())); return perm_[i]; }

      const intt& Pt(std::size_t i) const
      { assert(i<std::size_t(n())); return iperm_[i]; }
      intt& Pt(std::size_t i)
      { assert(i<std::size_t(n_)); return iperm_[i]; }

      const intt& operator[](std::size_t i) const
      { assert(i<std::size_t(n())); return perm_[i]; }
      intt& operator[](std::size_t i)
      { assert(i<std::size_t(n())); return perm_[i]; }

      void set_Pt();
      void set_P();

      void to_1_based();
      void to_0_based();

      PPt<intt>& identity();
      void random();

      bool valid() const;
      void check() const;
      void check_P() const;

    private:
      intt n_ = 0;
      std::vector<intt,NoInit<intt>> perm_, iperm_;

      static std::minstd_rand0 rgen_; // thread issues??
      bool valid(const intt* p) const;
      void check(const intt* p) const;
    };

    template<typename intt>
    std::minstd_rand0 PPt<intt>::rgen_ = std::minstd_rand0();

    template<typename intt> PPt<intt>& PPt<intt>::identity() {
      for (intt i=0; i<n_; i++) perm_[i] = i;
      for (intt i=0; i<n_; i++) iperm_[i] = i;
      return *this;
    }

    template<typename intt> void PPt<intt>::set_Pt() {
      assert(valid(P()));
      for (intt i=0; i<n_; i++) iperm_[perm_[i]] = i;
    }

    template<typename intt> void PPt<intt>::set_P() {
      assert(valid(Pt()));
      for (intt i=0; i<n_; i++) perm_[iperm_[i]] = i;
    }


    template<typename intt> void PPt<intt>::to_1_based() {
      for (intt i=0; i<n_; i++) perm_[i]++;
      for (intt i=0; i<n_; i++) iperm_[i]++;
    }

    template<typename intt> void PPt<intt>::to_0_based() {
      for (intt i=0; i<n_; i++) perm_[i]--;
      for (intt i=0; i<n_; i++) iperm_[i]--;
    }

    template<typename intt> bool PPt<intt>::valid() const {
      if (!valid(P())) return false;
      if (!valid(Pt())) return false;
      for (intt i=0; i<n_; i++)
        if (perm_[iperm_[i]] != i || iperm_[perm_[i]] != i)
          return false;
      return true;
    }

    template<typename intt> void PPt<intt>::check() const {
      check(perm_.data());
      check(iperm_.data());
      for (intt i=0; i<n_; i++) {
        assert(perm_[iperm_[i]] == i && iperm_[perm_[i]] == i);
      }
    }

    template<typename intt> void PPt<intt>::check_P() const {
      check(perm_.data());
    }

    template<typename intt> bool PPt<intt>::valid(const intt* p) const {
      std::vector<bool> mask(n_, false);
      for (intt i=0; i<n_; i++) {
        if (p[i] < 0 && p[i] >= n_) return false;
        mask[p[i]] = true;
      }
      for (intt i=0; i<n_; i++) if (!mask[i]) return false;
      return true;
    }

    template<typename intt> void PPt<intt>::check(const intt* p) const {
      std::vector<bool> mask(n_, false);
      for (intt i=0; i<n_; i++) {
        assert(p[i] >= 0 && p[i] < n_);
        mask[p[i]] = true;
      }
      for (intt i=0; i<n_; i++) {
        assert(mask[i] == true);
      }
    }

    template<typename intt> void PPt<intt>::random() {
      std::uniform_int_distribution<intt> uniform(0, n_-1);
      for (intt i=0; i<n_; i++) perm_[i] = i;
      for (intt i=0; i<n_; i++)
        std::swap(perm_[i], perm_[uniform(rgen_)]);
      for (intt i=0; i<n_; i++) iperm_[perm_[i]] = i;
    }

  } // end namespace ordering
} // end namespace strumpack

#endif // STRUMPACK_ORDERING_PPT_HPP
