#include <dram/dram_interface.h>
#include <dram/pim_kernel.h>

namespace llm_system {
namespace PIM_KERNEL {

void ModUp_Evkmult_kernel(PIMRequest& pim_request, DRAMRequestType dramreq_type,
                          DRAMRequest::PIM_Operand& operand,
                          const PIMHWConfig pim_hw_config) {
  // const int num_modup_buffer = 8;  // number of buffer, each holds 256-bit
  // const int num_accum_buffer = 8;  // twice for total accum buffer
  const int total_buffer = pim_hw_config.num_buffer;
  Ramulator::AddrVec_t addr_vec = {0, 0, 0, 0, 0, 0};

  auto precomputed = get_operand(operand, PIMOperandType::kPrecomputed);
  auto src = get_operand(operand, PIMOperandType::kSrc);
  auto dest = get_operand(operand, PIMOperandType::kDest);

  int num_dnum = src.size();
  int num_dest = dest.size();

  int num_chunk = (src[0]->GetAddress()).num_chunks_;

  // assertTrue(src.size() == 2, "ModUp_Evkmult operands are not valid");
  // assertTrue(num_dnum * 3 < total_buffer,
  //            "ModUp_Evkmult may results invalid value");
  assertTrue(precomputed.size() == 2 * num_dnum,
             "ModUp_Evkmult operands are not valid");
  assertTrue(dest.size() == 2, "ModUp_Evkmult destinations are not valid");

  // the number of chunks which can be buffered
  int num_chunk_in_buffer = total_buffer / (num_dnum + num_dest);
  int max_num_dnum_in_buffer = std::min(total_buffer - num_dest, num_dnum);
  num_chunk_in_buffer = std::max(num_chunk_in_buffer, 1);

  // loop for read Modup to RF
  for (int chunk_idx = 0; chunk_idx < num_chunk;
       chunk_idx += num_chunk_in_buffer) {
    for (int dnum_idx = 0; dnum_idx < num_dnum;
         dnum_idx += max_num_dnum_in_buffer) {
      for (int dnum_inner_idx = 0;
           dnum_inner_idx <
           std::min(max_num_dnum_in_buffer, num_dnum - dnum_idx);
           dnum_inner_idx++) {
        for (int rf_inner_idx = 0;
             rf_inner_idx <
             std::min(num_chunk_in_buffer, num_chunk - chunk_idx);
             rf_inner_idx++) {
          Limb::Ptr opnd = src[dnum_idx + dnum_inner_idx];
          addr_vec[4] = opnd->GetRowAddr(chunk_idx + rf_inner_idx);
          addr_vec[5] = opnd->GetColAddr(chunk_idx + rf_inner_idx);
          pim_request.AddCommand(PIMCommand(PIMCommandType::kDRAM2RF,
                                            PIMOperandType::kModUp, addr_vec,
                                            &pim_request, dramreq_type));
        }
      }
    }
    for (int dnum_idx = 0; dnum_idx < num_dnum;
         dnum_idx += max_num_dnum_in_buffer) {
      for (int dnum_inner_idx = 0;
           dnum_inner_idx <
           std::min(max_num_dnum_in_buffer, num_dnum - dnum_idx);
           dnum_inner_idx++) {
        for (int rf_inner_idx = 0;
             rf_inner_idx <
             std::min(num_chunk_in_buffer, num_chunk - chunk_idx);
             rf_inner_idx++) {
          Limb::Ptr opnd = precomputed[(dnum_idx + dnum_inner_idx) * 2];

          addr_vec[4] = opnd->GetRowAddr(chunk_idx + rf_inner_idx);
          addr_vec[5] = opnd->GetColAddr(chunk_idx + rf_inner_idx);
          pim_request.AddCommand(PIMCommand(PIMCommandType::kMAC,
                                            PIMOperandType::kEvk, addr_vec,
                                            &pim_request, dramreq_type));

          opnd = precomputed[dnum_inner_idx * 2 + 1];
          addr_vec[4] = opnd->GetRowAddr(chunk_idx + rf_inner_idx);
          addr_vec[5] = opnd->GetColAddr(chunk_idx + rf_inner_idx);
          pim_request.AddCommand(PIMCommand(PIMCommandType::kMAC,
                                            PIMOperandType::kEvk, addr_vec,
                                            &pim_request, dramreq_type));
        }
      }
    }
    for (auto des_opnd : dest) {
      for (int rf_inner_idx = 0;
           rf_inner_idx < std::min(num_chunk_in_buffer, num_chunk - chunk_idx);
           rf_inner_idx++) {
        addr_vec[4] = des_opnd->GetRowAddr(chunk_idx + rf_inner_idx);
        addr_vec[5] = des_opnd->GetColAddr(chunk_idx + rf_inner_idx);
        pim_request.AddCommand(PIMCommand(PIMCommandType::kRF2DRAM,
                                          PIMOperandType::kDest, addr_vec,
                                          &pim_request, dramreq_type));
      }
    }
  }
}

}  // namespace PIM_KERNEL
}  // namespace llm_system