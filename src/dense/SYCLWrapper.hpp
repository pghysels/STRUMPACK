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
#ifndef STRUMPACK_SYCL_WRAPPER_HPP
#define STRUMPACK_SYCL_WRAPPER_HPP

#include <cmath>
#include <complex>
#include <iostream>
#include <cassert>
#include <memory>
#include <map>

#if __has_include(<sycl/sycl.hpp>)
 #include <sycl/sycl.hpp>
#else
 #include <CL/sycl.hpp>
 namespace sycl = cl::sycl;
#endif
// #include <CL/sycl.hpp>
// #include "mkl.h"
#include "oneapi/mkl.hpp"

#include "GPUWrapper.hpp"


namespace strumpack {
  namespace gpu {

    auto async_handler = [](sycl::exception_list exceptions) {
      for (std::exception_ptr const &e : exceptions) {
        try {
          std::rethrow_exception(e);
        } catch (sycl::exception const &e) {
          std::cerr << "Caught asynchronous SYCL exception:" << std::endl
                    << e.what() << std::endl
                    << "Exception caught at file:" << __FILE__
                    << ", line:" << __LINE__ << std::endl;
        }
      }
    };

    class DeviceExt : public sycl::device {
    public:
      DeviceExt() : sycl::device() {}
      ~DeviceExt() { std::lock_guard<std::mutex> lock(m_mutex); }
      DeviceExt(const sycl::device& base) : sycl::device(base) {}

    private:
      mutable std::mutex m_mutex;
    };

    static inline int get_tid() {
      std::cout << "TODO get_tid" << std::endl;
      return 0;
      // return syscall(SYS_gettid);
    }

    class DeviceManager {
    public:
      int current_device() {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = _thread2dev_map.find(get_tid());
        if (it != _thread2dev_map.end()) {
          check_id(it->second);
          return it->second;
        }
        std::cerr
          << "WARNING: no SYCL device found in the map, returning DEFAULT_DEVICE_ID"
          << std::endl;
        return DEFAULT_DEVICE_ID;
      }
      sycl::queue* current_queue() {
        return _queues[current_device()];
      }

      void select_device(int id) {
        std::lock_guard<std::mutex> lock(m_mutex);
        check_id(id);
        _thread2dev_map[get_tid()] = id;
      }
      int device_count() { return _queues.size(); }

      /// Returns the instance of device manager singleton.
      static DeviceManager& instance() {
        static DeviceManager d_m;
        return d_m;
      }
      DeviceManager(const DeviceManager&)            = delete;
      DeviceManager& operator=(const DeviceManager&) = delete;
      DeviceManager(DeviceManager&&)                 = delete;
      DeviceManager& operator=(DeviceManager&&)      = delete;

    private:
      mutable std::mutex m_mutex;

      DeviceManager() {
        std::cout << "TODO what is sycl::gpu_selector_v" << std::endl;
        // sycl::device dev(sycl::gpu_selector_v);
        sycl::device dev(sycl::gpu_selector);
        _queues.push_back
          (new sycl::queue
           (dev, async_handler,
            sycl::property_list{sycl::property::queue::in_order{}}));
      }

      void check_id(int id) const {
        if (id >= _queues.size()) {
          throw std::runtime_error("invalid device id");
        }
      }

      // Note: only 1 out-of-order SYCL queue is created per device
      std::vector<sycl::queue*> _queues;

      /// DEFAULT_DEVICE_ID is used, if current_device() can not find
      /// current thread id in _thread2dev_map, which means default
      /// device should be used for the current thread.
      const int DEFAULT_DEVICE_ID = 0;
      /// thread-id to device-id map.
      std::map<int, int> _thread2dev_map;
    };

    /// Util function to get the current device (in int).
    static inline int get_sycl_device() {
      return DeviceManager::instance().current_device();
    }

    /// Util function to get the current queue
    static inline sycl::queue& get_sycl_queue() {
      return *(DeviceManager::instance().current_queue());
    }

    const sycl::queue& get_sycl_queue(const Stream& s);
    sycl::queue& get_sycl_queue(Stream& s);

  } // end namespace sycl
} // end namespace strumpack

#endif // STRUMPACK_SYCL_WRAPPER_HPP
