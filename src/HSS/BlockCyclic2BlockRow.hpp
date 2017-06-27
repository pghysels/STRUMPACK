#ifndef BLOCK_CYCLIC_2_BLOCK_ROW_HPP
#define BLOCK_CYCLIC_2_BLOCK_ROW_HPP

namespace strumpack {
  namespace HSS {
    namespace BC2BR {

      template<typename scalar_t> void block_cyclic_to_block_row
      (const TreeLocalRanges& ranges, const DistributedMatrix<scalar_t>& dist,
       DenseMatrix<scalar_t>& sub, DistributedMatrix<scalar_t>& leaf, int lctxt,
       MPI_Comm comm) {
	assert(dist.fixed());
	const auto P = mpi_nprocs(comm);
	const auto rank = mpi_rank(comm);
	std::vector<std::vector<Triplet<scalar_t>>> sbuf(P);

	const int MB = DistributedMatrix<scalar_t>::default_MB;
	auto d = dist.cols();
	for (int p=0; p<P; p++) {
	  auto m = ranges.chi(p) - ranges.clo(p);
	  auto leaf_procs = ranges.leaf_procs(p);
	  auto rbegin = ranges.clo(p) - ranges.clo(0);
	  auto pdist = ConstDistributedMatrixWrapperPtr(m, d, dist, rbegin, 0);
	  int rlo, rhi, clo, chi;
	  pdist->lranges(rlo, rhi, clo, chi);
	  if (leaf_procs == 1) {
	    if (p == rank) {
	      sub = DenseMatrix<scalar_t>(m, d);
	      if (dist.active()) {
		for (int c=clo; c<chi; c++) {
		  auto gc = dist.coll2g_fixed(c);
		  for (int r=rlo; r<rhi; r++)
		    sub(dist.rowl2g_fixed(r) - rbegin, gc) = dist(r,c);
		}
	      }
	    } else {
	      if (dist.active()) {
		for (int c=clo; c<chi; c++) {
		  auto gc = dist.coll2g_fixed(c);
		  for (int r=rlo; r<rhi; r++)
		    sbuf[p].emplace_back(dist.rowl2g_fixed(r) - rbegin, gc, dist(r,c));
		}
	      }
	    }
	  } else {
	    if (p <= rank && rank < p+leaf_procs) leaf = DistributedMatrix<scalar_t>(lctxt, m, d);
	    if (dist.active()) {
	      const int leaf_pcols = std::floor(std::sqrt((float)leaf_procs));
	      const int leaf_prows = leaf_procs / leaf_pcols;
	      for (int c=clo; c<chi; c++) {
		auto gc = dist.coll2g_fixed(c);
		for (int r=rlo; r<rhi; r++) {
		  auto gr = dist.rowl2g_fixed(r) - rbegin;
		  std::size_t dest = p + ((gr / MB) % leaf_prows) + ((gc / MB) % leaf_pcols) * leaf_prows;
		  sbuf[dest].emplace_back(gr, gc, dist(r,c));
		}
	      }
	    }
	    p += ranges.leaf_procs(p)-1;
	  }
	}
	MPI_Datatype triplet_type;
	create_triplet_mpi_type<scalar_t>(&triplet_type);
	Triplet<scalar_t>* recvbuf;
	std::size_t totrsize;
	all_to_all_v(sbuf, recvbuf, totrsize, comm, triplet_type);
	MPI_Type_free(&triplet_type);
	if (ranges.leaf_procs(rank) == 1)
	  for (auto t=recvbuf; t!=recvbuf+totrsize; t++)
	    sub(t->_r, t->_c) = t->_v;
	else
	  if (leaf.active())
	    for (auto t=recvbuf; t!=recvbuf+totrsize; t++)
	      leaf.global_fixed(t->_r, t->_c) = t->_v;
	delete[] recvbuf;
      }

      template<typename scalar_t> void block_row_to_block_cyclic
      (const TreeLocalRanges& ranges, DistributedMatrix<scalar_t>& dist,
       const DenseMatrix<scalar_t>& sub, const DistributedMatrix<scalar_t>& leaf, MPI_Comm comm) {
	assert(dist.fixed());
	const auto P = mpi_nprocs(comm);
	const auto rank = mpi_rank(comm);
	const int MB = DistributedMatrix<scalar_t>::default_MB;
	const int dist_pcols = std::floor(std::sqrt((float)P));
	const int dist_prows = P / dist_pcols;
	const auto leaf_procs = ranges.leaf_procs(rank);
	auto rbegin = ranges.clo(rank) - ranges.clo(0);
	std::vector<std::vector<Triplet<scalar_t>>> sbuf(P);
	if (leaf_procs == 1) { // sub
	  for (std::size_t c=0; c<sub.cols(); c++)
	    for (std::size_t r=0; r<sub.rows(); r++) {
	      auto gr = r + rbegin;
	      std::size_t dest = ((gr / MB) % dist_prows) + ((c / MB) % dist_pcols) * dist_prows;
	      sbuf[dest].emplace_back(gr, c, sub(r, c));
	    }
	} else { // leaf
	  if (leaf.active())
	    for (int c=0; c<leaf.lcols(); c++) {
	      auto gc = leaf.coll2g_fixed(c);
	      for (int r=0; r<leaf.lrows(); r++) {
		auto gr = leaf.rowl2g_fixed(r) + rbegin;
		std::size_t dest = ((gr / MB) % dist_prows) + ((gc / MB) % dist_pcols) * dist_prows;
		sbuf[dest].emplace_back(gr, gc, leaf(r, c));
	      }
	    }
	}
	MPI_Datatype triplet_type;
	create_triplet_mpi_type<scalar_t>(&triplet_type);
	Triplet<scalar_t>* recvbuf;
	std::size_t totrsize;
	all_to_all_v(sbuf, recvbuf, totrsize, comm, triplet_type);
	MPI_Type_free(&triplet_type);
	if (dist.active())
	  for (auto t=recvbuf; t!=recvbuf+totrsize; t++)
	    dist.global_fixed(t->_r, t->_c) = t->_v;
	delete[] recvbuf;
      }

    } //end namespace BC2BR
  } // end namespace HSS
} //end namespace strumpack

#endif // BLOCK_CYCLIC_2_BLOCK_ROW_HPP
