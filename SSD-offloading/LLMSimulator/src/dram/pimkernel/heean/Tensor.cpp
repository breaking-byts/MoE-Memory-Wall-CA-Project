#include <dram/dram_interface.h>
#include <dram/pim_kernel.h>

namespace llm_system {
namespace PIM_KERNEL {

void Tensor_kernel(PIMRequest& pim_request, DRAMRequestType dramreq_type,
                   DRAMRequest::PIM_Operand& operand,
                   const PIMHWConfig pim_hw_config) {
  const int total_buffer = pim_hw_config.num_buffer;

  Ramulator::AddrVec_t addr_vec = {0, 0, 0, 0, 0, 0};

  auto src = get_operand(operand, PIMOperandType::kSrc);
  auto dest = get_operand(operand, PIMOperandType::kDest);

  int num_src = src.size();
  int num_dest = dest.size();

  int num_chunk = (dest[0]->GetAddress()).num_chunks_;

  // the number of chunks which can be buffered
  // a1, a2, b2, a, b, aux is buffered
  // input limbs : (b1, a1, b2, a2)
  const int num_chunk_in_buffer = total_buffer / (num_src + num_dest - 1);

  assertTrue(src.size() == 4, "Tensor operands are not valid");
  assertTrue(dest.size() == 3, "Tensor destinations are not valid");

  for (int chunk_idx = 0; chunk_idx < num_chunk;
       chunk_idx += num_chunk_in_buffer) {
    // read a1
    // read a2
    // read b2
    // a = a1 * b2 (read b2 from DRAM)
    // a += b1 * a2 (read b1 from DRAM)
    // b = b1 * b2 (read b1 from DRAM)
    // aux = a1 * a2 (no read from DRAM)
    // write a, b, aux to DRAM

    // read a1
    for (int rf_inner_idx = 0;
         rf_inner_idx < std::min(num_chunk_in_buffer, num_chunk - chunk_idx);
         rf_inner_idx++) {
      Limb::Ptr src_opnd = src[1];
      addr_vec[4] = src_opnd->GetRowAddr(chunk_idx + rf_inner_idx);
      addr_vec[5] = src_opnd->GetColAddr(chunk_idx + rf_inner_idx);
      pim_request.AddCommand(PIMCommand(PIMCommandType::kDRAM2RF,
                                        PIMOperandType::kSrc, addr_vec,
                                        &pim_request, dramreq_type));
    }
    // read a2
    for (int rf_inner_idx = 0;
         rf_inner_idx < std::min(num_chunk_in_buffer, num_chunk - chunk_idx);
         rf_inner_idx++) {
      Limb::Ptr src_opnd = src[3];
      addr_vec[4] = src_opnd->GetRowAddr(chunk_idx + rf_inner_idx);
      addr_vec[5] = src_opnd->GetColAddr(chunk_idx + rf_inner_idx);
      pim_request.AddCommand(PIMCommand(PIMCommandType::kDRAM2RF,
                                        PIMOperandType::kSrc, addr_vec,
                                        &pim_request, dramreq_type));
    }
    // read b2
    for (int rf_inner_idx = 0;
         rf_inner_idx < std::min(num_chunk_in_buffer, num_chunk - chunk_idx);
         rf_inner_idx++) {
      Limb::Ptr src_opnd = src[2];
      addr_vec[4] = src_opnd->GetRowAddr(chunk_idx + rf_inner_idx);
      addr_vec[5] = src_opnd->GetColAddr(chunk_idx + rf_inner_idx);
      pim_request.AddCommand(PIMCommand(PIMCommandType::kDRAM2RF,
                                        PIMOperandType::kSrc, addr_vec,
                                        &pim_request, dramreq_type));
    }
    // a = a1 * b2 (read b2 from DRAM)
    for (int rf_inner_idx = 0;
         rf_inner_idx < std::min(num_chunk_in_buffer, num_chunk - chunk_idx);
         rf_inner_idx++) {
      Limb::Ptr src_opnd = src[2];
      addr_vec[4] = src_opnd->GetRowAddr(chunk_idx + rf_inner_idx);
      addr_vec[5] = src_opnd->GetColAddr(chunk_idx + rf_inner_idx);
      pim_request.AddCommand(PIMCommand(PIMCommandType::kMult,
                                        PIMOperandType::kSrc, addr_vec,
                                        &pim_request, dramreq_type));
    }
    // a += b1 * a2 (read b1 from DRAM)
    for (int rf_inner_idx = 0;
         rf_inner_idx < std::min(num_chunk_in_buffer, num_chunk - chunk_idx);
         rf_inner_idx++) {
      Limb::Ptr src_opnd = src[0];
      addr_vec[4] = src_opnd->GetRowAddr(chunk_idx + rf_inner_idx);
      addr_vec[5] = src_opnd->GetColAddr(chunk_idx + rf_inner_idx);
      pim_request.AddCommand(PIMCommand(PIMCommandType::kMAC,
                                        PIMOperandType::kSrc, addr_vec,
                                        &pim_request, dramreq_type));
    }
    // b = b1*b2 (read b1 from DRAM)
    for (int rf_inner_idx = 0;
         rf_inner_idx < std::min(num_chunk_in_buffer, num_chunk - chunk_idx);
         rf_inner_idx++) {
      Limb::Ptr src_opnd = src[0];
      addr_vec[4] = src_opnd->GetRowAddr(chunk_idx + rf_inner_idx);
      addr_vec[5] = src_opnd->GetColAddr(chunk_idx + rf_inner_idx);
      pim_request.AddCommand(PIMCommand(PIMCommandType::kMult,
                                        PIMOperandType::kSrc, addr_vec,
                                        &pim_request, dramreq_type));
    }
    // aux = a1 * a2 (no read from DRAM, just dummy)
    for (int rf_inner_idx = 0;
         rf_inner_idx < std::min(num_chunk_in_buffer, num_chunk - chunk_idx);
         rf_inner_idx++) {
      Limb::Ptr src_opnd = src[0];
      addr_vec[4] = src_opnd->GetRowAddr(chunk_idx + rf_inner_idx);
      addr_vec[5] = src_opnd->GetColAddr(chunk_idx + rf_inner_idx);
      pim_request.AddCommand(PIMCommand(PIMCommandType::kMult,
                                        PIMOperandType::kSrc, addr_vec,
                                        &pim_request, dramreq_type));
    }
    // write a, b, aux to DRAM
    for (auto dest_opnd : dest) {
      for (int rf_inner_idx = 0;
           rf_inner_idx < std::min(num_chunk_in_buffer, num_chunk - chunk_idx);
           rf_inner_idx++) {
        addr_vec[4] = dest_opnd->GetRowAddr(chunk_idx + rf_inner_idx);
        addr_vec[5] = dest_opnd->GetColAddr(chunk_idx + rf_inner_idx);
        pim_request.AddCommand(PIMCommand(PIMCommandType::kRF2DRAM,
                                          PIMOperandType::kRF, addr_vec,
                                          &pim_request, dramreq_type));
      }
    }
  }
}

}  // namespace PIM_KERNEL
}  // namespace llm_system