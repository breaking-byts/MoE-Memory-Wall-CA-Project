#include <dram/dram_interface.h>
#include <dram/pim_kernel.h>

namespace llm_system {
namespace PIM_KERNEL {

void CMAD_kernel(PIMRequest& pim_request, DRAMRequestType dramreq_type,
                 DRAMRequest::PIM_Operand& operand,
                 const PIMHWConfig pim_hw_config) {
  const int total_buffer = pim_hw_config.num_buffer;

  Ramulator::AddrVec_t addr_vec = {0, 0, 0, 0, 0, 0};

  auto src = get_operand(operand, PIMOperandType::kSrc);
  auto dest = get_operand(operand, PIMOperandType::kDest);

  int dnum = src.size();
  int evk_size = dest.size();

  int mde_chunk = (dest[0]->GetAddress()).num_chunks_;

  // the number of chunks which can be buffered
  const int num_chunk_in_buffer = total_buffer;

  assertTrue(src.size() == 2, "CMAD operands are not valid");
  assertTrue(dest.size() == 1, "Mult destinations are not valid");

  // loop for read Modup to RF
  for (int chunk_idx = 0; chunk_idx < mde_chunk;
       chunk_idx += num_chunk_in_buffer) {
    // b = a1 * c
    // b += a2
    for (auto src_opnd : src) {
      for (int rf_inner_idx = 0;
           rf_inner_idx < std::min(num_chunk_in_buffer, mde_chunk - chunk_idx);
           rf_inner_idx++) {
        addr_vec[4] = src_opnd->GetRowAddr(chunk_idx + rf_inner_idx);
        addr_vec[5] = src_opnd->GetColAddr(chunk_idx + rf_inner_idx);
        pim_request.AddCommand(PIMCommand(PIMCommandType::kMAC,
                                          PIMOperandType::kSrc, addr_vec,
                                          &pim_request, dramreq_type));
      }
    }
    // write b to DRAM
    for (auto dest_opnd : dest) {
      for (int rf_inner_idx = 0;
           rf_inner_idx < std::min(num_chunk_in_buffer, mde_chunk - chunk_idx);
           rf_inner_idx++) {
        addr_vec[4] = dest_opnd->GetRowAddr(chunk_idx + rf_inner_idx);
        addr_vec[5] = dest_opnd->GetColAddr(chunk_idx + rf_inner_idx);
        pim_request.AddCommand(PIMCommand(PIMCommandType::kDRAM2RF,
                                          PIMOperandType::kRF, addr_vec,
                                          &pim_request, dramreq_type));
      }
    }
  }
}

}  // namespace PIM_KERNEL
}  // namespace llm_system