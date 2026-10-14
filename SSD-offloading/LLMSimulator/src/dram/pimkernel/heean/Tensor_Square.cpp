#include <dram/dram_interface.h>
#include <dram/pim_kernel.h>

namespace llm_system {
namespace PIM_KERNEL {

void Tensor_Square_kernel(PIMRequest& pim_request, DRAMRequestType dramreq_type,
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
  // b1, a1 is buffered
  // input limbs : (b1, a1)
  const int num_chunk_in_buffer = total_buffer / (num_src + num_dest);

  assertTrue(src.size() == 2, "Tensor_Square operands are not valid");
  assertTrue(dest.size() == 3, "Tensor_Square destinations are not valid");

  for (int chunk_idx = 0; chunk_idx < num_chunk;
       chunk_idx += num_chunk_in_buffer) {
    // read a1
    // aux = a1 * a1 (read a1 from DRAM)
    // read b1
    // b = b1 * b1 (read b1 from DRAM)
    // a = 2 * b1 (read b1 from DRAM)
    // a *= a1 (no read from DRAM)
    // write a, b, aux to DRAM

    // read a1
    // aux = a1 * a1 (read a1 from DRAM)
    for (int rf_inner_idx = 0;
         rf_inner_idx < std::min(num_chunk_in_buffer, num_chunk - chunk_idx);
         rf_inner_idx++) {
      Limb::Ptr src_opnd = src[1];
      addr_vec[4] = src_opnd->GetRowAddr(chunk_idx + rf_inner_idx);
      addr_vec[5] = src_opnd->GetColAddr(chunk_idx + rf_inner_idx);
      pim_request.AddCommand(PIMCommand(PIMCommandType::kDRAM2RF,
                                        PIMOperandType::kSrc, addr_vec,
                                        &pim_request, dramreq_type));
      pim_request.AddCommand(PIMCommand(PIMCommandType::kMult,
                                        PIMOperandType::kSrc, addr_vec,
                                        &pim_request, dramreq_type));
    }
    // read b1
    // b = b1 * b1 (read b1 from DRAM)
    // a = 2 * b1 (read b1 from DRAM)
    // a *= a1 (no read from DRAM)
    for (int rf_inner_idx = 0;
         rf_inner_idx < std::min(num_chunk_in_buffer, num_chunk - chunk_idx);
         rf_inner_idx++) {
      Limb::Ptr src_opnd = src[0];
      addr_vec[4] = src_opnd->GetRowAddr(chunk_idx + rf_inner_idx);
      addr_vec[5] = src_opnd->GetColAddr(chunk_idx + rf_inner_idx);
      pim_request.AddCommand(PIMCommand(PIMCommandType::kDRAM2RF,
                                        PIMOperandType::kSrc, addr_vec,
                                        &pim_request, dramreq_type));
      pim_request.AddCommand(PIMCommand(PIMCommandType::kMult,
                                        PIMOperandType::kSrc, addr_vec,
                                        &pim_request, dramreq_type));
      pim_request.AddCommand(PIMCommand(PIMCommandType::kMult,
                                        PIMOperandType::kSrc, addr_vec,
                                        &pim_request, dramreq_type));
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