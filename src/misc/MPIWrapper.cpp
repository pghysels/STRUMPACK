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
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cassert>
#if defined(_OPENMP)
#include <omp.h>
#endif
#include "MPIWrapper.hpp"

using namespace strumpack;

MessageList Message::message_log_list = MessageList();

Message::Message(MsgType type, std::size_t size)
  : size_(size), type_(type) {
  id_ = message_log_list.list.size()+1;
  message_log_list.list.push_back(*this);
}

Message::~Message() {}

void Message::print(std::ostream& os) {
  MPIComm c;
  int rank = c.rank();
  os <<  "#   - size of ";
  switch (type_) {
  case MsgType::BROADCAST:       os << "broadcast"; break;
  case MsgType::ALL_GATHER:      os << "all_gather"; break;
  case MsgType::GATHER:          os << "gather"; break;
  case MsgType::SEND:            os << "send"; break;
  case MsgType::ALL_REDUCE:      os << "allreduce"; break;
  case MsgType::REDUCE:          os << "reduce"; break;
  case MsgType::ALLTOALL:        os << "alltoall"; break;
  }
  os << " = " << size_
     << ", id = " << id_
     << " , rank: " << rank << "\n";
}

MessageList::MessageList() : is_finalized(false) {}

void MessageList::Finalize() {
  Message::message_log_list.finalize();
}

void MessageList::finalize() {
#if defined(STRUMPACK_MESSAGE_COUNTER)
  std::ofstream log;
  log.open("message.log", std::ofstream::out);
  for (auto msg : list)
    msg.print(log);
  return;
#endif
}
