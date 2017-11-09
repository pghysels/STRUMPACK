/*
 * STRUMPACK -- STRUctured Matrices PACKage, Copyright (c) 2014, The Regents of
 * the University of California, through Lawrence Berkeley National Laboratory
 * (subject to receipt of any required approvals from the U.S. Dept. of Energy).
 * All rights reserved.
 *
 * If you have questions about your rights to use or distribute this software,
 * please contact Berkeley Lab's Technology Transfer Department at TTD@lbl.gov.
 *
 * NOTICE. This software is owned by the U.S. Department of Energy. As such, the
 * U.S. Government has been granted for itself and others acting on its behalf a
 * paid-up, nonexclusive, irrevocable, worldwide license in the Software to
 * reproduce, prepare derivative works, and perform publicly and display publicly.
 * Beginning five (5) years after the date permission to assert copyright is
 * obtained from the U.S. Department of Energy, and subject to any subsequent five
 * (5) year renewals, the U.S. Government is granted for itself and others acting
 * on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
 * Software to reproduce, prepare derivative works, distribute copies to the
 * public, perform publicly and display publicly, and to permit others to do so.
 *
 * Developers: Pieter Ghysels, Francois-Henry Rouet, Xiaoye S. Li.
 *             (Lawrence Berkeley National Lab, Computational Research Division).
 *
 */
#ifndef CSRGRAPH_HPP
#define CSRGRAPH_HPP
#include <sstream>
#include <fstream>
#include <vector>
#include <tuple>
#include <unordered_map>
#include "misc/MPIWrapper.hpp"

namespace strumpack {

  template<typename integer_t> class CSRGraph {
  public:
    CSRGraph();
    CSRGraph(const CSRGraph&) = delete;
    CSRGraph& operator=(const CSRGraph&) = delete;
    CSRGraph(CSRGraph&&);
    CSRGraph& operator=(CSRGraph&&);
    CSRGraph(integer_t n, integer_t nvert);
    ~CSRGraph();
    void print();
    inline integer_t size() const { return _n_vert; }
    inline integer_t nvert() const { return _n_vert; }
    inline integer_t nedge() const { return _n_edge; }
    inline integer_t* get_ptr() const { return _ptr; }
    inline integer_t* get_ind() const { return _ind; }
    void sort_rows();
    void permute_local(integer_t* order, integer_t* iorder, integer_t clo, integer_t chi);
    void permute_columns(integer_t* order);
    void permute_rows_local_cols_global(integer_t* order, integer_t* iorder, integer_t clo, integer_t chi);
    void extract_separator_subgraph(int sep_order_level, integer_t lo, integer_t sep_begin, integer_t sep_end,
				    integer_t part, integer_t* order, std::vector<integer_t>& sep_csr_ptr,
				    std::vector<integer_t>& sep_csr_ind);
    void clear_temp_data() { _o.clear(); }

  private:
    integer_t _n_vert;
    integer_t _n_edge;
    integer_t* _ptr;
    integer_t* _ind;

    /* This (unordered_)map is used to find length-2 edges that go
       outside the local graph. This avoids a quadratic search */
    std::unordered_map<integer_t, std::vector<integer_t>> _o;
  };

  template<typename integer_t>
  CSRGraph<integer_t>::CSRGraph() : _n_vert(0), _n_edge(0), _ptr(NULL), _ind(NULL) {}

  template<typename integer_t>
  CSRGraph<integer_t>::CSRGraph(integer_t nr_vert, integer_t nr_edge)
    : _n_vert(nr_vert), _n_edge(nr_edge) {
    _ptr = new integer_t[nr_vert+1];
    _ind = new integer_t[nr_edge];
  }

  template<typename integer_t>
  CSRGraph<integer_t>::CSRGraph(CSRGraph&& rhs) {
    _n_vert = rhs._n_vert;
    _n_edge = rhs._n_edge;
    _ptr = rhs._ptr;
    _ind = rhs._ind;
    rhs._ptr = rhs._ind = nullptr;
  }

  template<typename integer_t> CSRGraph<integer_t>&
  CSRGraph<integer_t>::operator=(CSRGraph<integer_t> &&rhs) {
    if (this != &rhs) {
      _n_vert = rhs._n_vert;
      _n_edge = rhs._n_edge;
      rhs._n_vert = rhs._n_edge = 0;
      delete[] _ptr;
      delete[] _ind;
      _ptr = rhs._ptr;
      _ind = rhs._ind;
      rhs._ptr = rhs._ind = nullptr;
    }
    return *this;
  }

  template<typename integer_t> CSRGraph<integer_t>::~CSRGraph() {
    delete[] _ptr;
    delete[] _ind;
  }

  template<typename integer_t> void
  CSRGraph<integer_t>::sort_rows() {
#pragma omp parallel for
    for (integer_t r=0; r<_n_vert; r++)
      std::sort(_ind+_ptr[r], _ind+_ptr[r+1]);
  }

  template<typename integer_t> void
  CSRGraph<integer_t>::print() {
    for (integer_t i=0; i<_n_vert; i++) {
      std::cout << "r=" << i << ", ";
      for (integer_t j=_ptr[i]; j<_ptr[i+1]; j++)
	std::cout << _ind[j] << " ";
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  /**
   * order and iorder are of size this->size() == this->n_vert() == chi-clo.
   * order has elements in the range [clo, chi).
   * iorder had elements in the range [0, chi-clo).
   *
   * This applies permutation only in the diagonal block. This is
   * useful for the local subgraphs. The off-diagonal elements
   * represent the connections to the distributed separators, which
   * are not permuted, so this permutation can be done completely
   * local!
   */
  template<typename integer_t> void
  CSRGraph<integer_t>::permute_local(integer_t* order, integer_t* iorder, integer_t clo, integer_t chi) {
    auto new_ptr = new integer_t[_n_vert+1];
    auto new_ind = new integer_t[_n_edge];
    integer_t nnz = 0;
    new_ptr[0] = 0;
    for (integer_t i=0; i<_n_vert; i++) {
      auto lb = _ptr[iorder[i]];
      auto ub = _ptr[iorder[i]+1];
      for (integer_t j=lb; j<ub; j++) {
	auto c = _ind[j];
	new_ind[nnz++] = (c >= clo && c < chi) ? order[c-clo] : c;
      }
      new_ptr[i+1] = nnz;
    }
    delete[] _ptr; _ptr = new_ptr;
    delete[] _ind; _ind = new_ind;
  }

  /**
   * iorder is of size this->size() == this->n_vert().
   * iorder had elements in the range [0, chi-clo).
   * order is of the global size.
   */
  template<typename integer_t> void CSRGraph<integer_t>::permute_rows_local_cols_global
  (integer_t* order, integer_t* iorder, integer_t clo, integer_t chi) {
    auto new_ptr = new integer_t[_n_vert+1];
    auto new_ind = new integer_t[_n_edge];
    integer_t nnz = 0;
    new_ptr[0] = 0;
    for (integer_t i=0; i<_n_vert; i++) {
      auto lb = _ptr[iorder[i]];
      auto ub = _ptr[iorder[i]+1];
      for (integer_t j=lb; j<ub; j++)
	new_ind[nnz++] = order[_ind[j]];
      new_ptr[i+1] = nnz;
    }
    delete[] _ptr; _ptr = new_ptr;
    delete[] _ind; _ind = new_ind;
  }

  template<typename integer_t> void CSRGraph<integer_t>::permute_columns(integer_t* order) {
    for (integer_t i=0; i<_n_vert; i++)
      for (integer_t j=_ptr[i]; j<_ptr[i+1]; j++)
	_ind[j] = order[_ind[j]];
  }

  template<typename integer_t> void
  CSRGraph<integer_t>::extract_separator_subgraph
  (int sep_order_level, integer_t lo, integer_t sep_begin, integer_t sep_end,
   integer_t part, integer_t* order, std::vector<integer_t>& sep_csr_ptr,
   std::vector<integer_t>& sep_csr_ind) {
    assert(sep_order_level == 0 || sep_order_level == 1);

    // for all nodes not in this graph to which there is an outgoing
    //   edge, we keep a list of edges to/from that node
    if (_o.empty()) {
      auto hi = lo + _n_vert;
      for (integer_t r=0; r<_n_vert; r++)
	for (integer_t j=_ptr[r]; j<_ptr[r+1]; j++) {
	  auto c = _ind[j];
	  if (c < lo || c >= hi) _o[c].push_back(r);
	}
    }

    sep_csr_ptr.clear();
    sep_csr_ind.clear();
    auto dim_sep = sep_end - sep_begin;
    std::vector<bool> mark(dim_sep);
    auto ind_to_part = new integer_t[dim_sep];
    integer_t count = 0;
    for (integer_t r=0; r<dim_sep; r++)
      ind_to_part[r] = (order[r] == part) ? count++ : -1;
    sep_csr_ptr.reserve(count+1);

    for (integer_t r=sep_begin, sep_edges=0; r<sep_end; r++) {
      if (order[r-sep_begin] == part) {
	sep_csr_ptr.push_back(sep_edges);
	std::fill(mark.begin(), mark.end(), false);
	for (integer_t j=_ptr[r]; j<_ptr[r+1]; j++) {
	  auto c = _ind[j] - lo;
	  if (c == r) continue;
	  if (c >= 0 && c < _n_vert) {
	    auto lc = c - sep_begin;
	    if (lc >= 0 && lc < dim_sep && order[lc] == part && !mark[lc]) {
	      mark[lc] = true;
	      sep_csr_ind.push_back(ind_to_part[lc]);
	      sep_edges++;
	    } else {
	      if (sep_order_level > 0) {
		for (integer_t k=_ptr[c]; k<_ptr[c+1]; k++) {
		  auto cc = _ind[k] - lo;
		  auto lcc = cc - sep_begin;
		  if (cc != r && lcc >= 0 && lcc < dim_sep && order[lcc] == part && !mark[lcc]) {
		    mark[lcc] = true;
		    sep_csr_ind.push_back(ind_to_part[lcc]);
		    sep_edges++;
		  }
		}
	      }
	    }
	  } else {
	    if (sep_order_level > 0) {
	      for (auto cc : _o[c+lo]) {
	      	auto lcc = cc - sep_begin;
	      	if (cc != r && lcc >= 0 && lcc < dim_sep && order[lcc] == part && !mark[lcc]) {
	      	  mark[lcc] = true;
	      	  sep_csr_ind.push_back(ind_to_part[lcc]);
	      	  sep_edges++;
	      	}
	      }
	    }
	  }
	}
      }
    }
    sep_csr_ptr.push_back(sep_csr_ind.size());
    delete[] ind_to_part;
  }

} // end namespace strumpack

#endif //CSRGRAPH_HPP
