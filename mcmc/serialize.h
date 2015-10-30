#ifndef __MCMC_SERIALIZE_H__
#define __MCMC_SERIALIZE_H__

#include <fstream>
#include <google/protobuf/message.h>

#include "protos.pb.h"
#include "mcmc/types.h"
#include "mcmc/partitioned-alloc.h"

namespace mcmc {

template <class MessageType>
bool SerializeMessage(std::ostream* out, const MessageType& message) {
  uint64_t byte_size = message.ByteSize();
  std::vector<char> buf(byte_size);
  if (!message.SerializeToArray(buf.data(), byte_size)) {
    LOG(DFATAL) << "Failure here";
    return false;
  }
  out->write((char*)&byte_size, sizeof(byte_size));
  out->write(buf.data(), byte_size);
  return true;
}

template <class MessageType>
bool ParseMessage(std::istream* in, MessageType* message) {
  uint64_t byte_size;
  in->read((char*)&byte_size, sizeof(byte_size));
  std::vector<char> buf(byte_size);
  in->read(buf.data(), byte_size);
  if (message->ParseFromArray(buf.data(), byte_size)) {
    return true;
  } else {
    LOG(DFATAL) << "Failure here";
    return false;
  }
}

template <class T>
bool Serialize(std::ostream* out, clcuda::Buffer<T>* buf,
               clcuda::Queue* queue) {
  VectorStorage vec;
  vec.mutable_storage()->resize(buf->GetSize());
  buf->Read(*queue, buf->GetSize() / sizeof(T),
            (T*)vec.mutable_storage()->data());
  if (SerializeMessage(out, vec)) {
    return true;
  } else {
    LOG(DFATAL) << "Failure here";
    return false;
  }
}

template <class T>
bool Parse(std::istream* in, clcuda::Buffer<T>* buf, clcuda::Queue* queue) {
  VectorStorage vec;
  if (!ParseMessage(in, &vec)) {
    LOG(DFATAL) << "Failure here";
    return false;
  }
  if (vec.storage().size() == buf->GetSize()) {
    buf->Write(*queue, vec.storage().size() / sizeof(T),
               (T*)vec.storage().data());
    return true;
  } else {
    LOG(DFATAL) << "Failure here";
    return false;
  }
}

template <class T>
bool Serialize(std::ostream* out, RowPartitionedMatrix<T>* rpm,
               clcuda::Queue* queue) {
  RpmProperties props;
  props.set_rows(rpm->Rows());
  props.set_cols(rpm->Cols());
  props.set_rows_in_block(rpm->RowsPerBlock());
  if (!SerializeMessage(out, props)) {
    LOG(DFATAL) << "Failure here";
    return false;
  }
  for (uint32_t i = 0; i < rpm->Blocks().size(); ++i) {
    if (!Serialize(out, &rpm->Blocks()[i], queue)) {
      LOG(DFATAL) << "Failure here";
      return false;
    }
  }
  return true;
}

template <class T>
bool Parse(std::istream* in, RowPartitionedMatrix<T>* rpm,
           clcuda::Queue* queue) {
  RpmProperties props;
  if (!ParseMessage(in, &props)) {
    LOG(DFATAL) << "Failure here";
    return false;
  }
  if (props.rows() != rpm->Rows() || props.cols() != rpm->Cols() ||
      props.rows_in_block() != rpm->RowsPerBlock()) {
    LOG(DFATAL) << "Failure here";
    return false;
  }
  for (uint32_t i = 0; i < props.rows(); i += props.rows_in_block()) {
    uint32_t idx = i / props.rows_in_block();
    if (!Parse(in, &rpm->Blocks()[idx], queue)) {
      LOG(DFATAL) << "Failure here";
      return false;
    }
  }
  return true;
}

}  // namespace mcmc

#endif  // __MCMC_SERIALIZE_H__
