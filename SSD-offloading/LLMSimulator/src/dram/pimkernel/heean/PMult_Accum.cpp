#include <dram/dram_interface.h>
#include <dram/pim_kernel.h>

namespace llm_system {
namespace PIM_KERNEL {

void PMult_Accum_kernel(PIMRequest& pim_request, DRAMRequestType dramreq_type,
                        DRAMRequest::PIM_Operand& operand,
                        const PIMHWConfig pim_hw_config) {
  const int total_buffer = pim_hw_config.num_buffer;
  Ramulator::AddrVec_t addr_vec = {0, 0, 0, 0, 0, 0};

  auto src = get_operand(operand, PIMOperandType::kSrc);
  auto precomputed = get_operand(operand, PIMOperandType::kPrecomputed);
  auto dest = get_operand(operand, PIMOperandType::kDest);

  int num_x = precomputed.size();
  int num_dest = dest.size();

  int num_chunk = (src[0]->GetAddress()).num_chunks_;

  // assertTrue(src.size() == 2, "PMult Accum operands are not valid");
  assertTrue(src.size() == 2 * precomputed.size(),
             "PMult Accum operands are not valid");
  assertTrue(dest.size() == 2, "PMult Accum destinations are not valid");

  // the number of chunks which can be buffered
  int num_chunk_in_buffer = total_buffer / (num_x + num_dest);
  int max_num_x_in_buffer = std::min(total_buffer - num_dest, num_x);
  num_chunk_in_buffer = std::max(num_chunk_in_buffer, 1);

  // std::next(buffer.begin(), std::min(max_size, buffer.size())
  // loop for read Modup to RF
  for (int chunk_idx = 0; chunk_idx < num_chunk;
       chunk_idx += num_chunk_in_buffer) {
    for (int x_idx = 0; x_idx < num_x; x_idx += max_num_x_in_buffer) {
      // read precomputed to RF
      for (int x_inner_idx = 0;
           x_inner_idx < std::min(max_num_x_in_buffer, num_x - x_idx);
           x_inner_idx++) {
        for (int rf_inner_idx = 0;
             rf_inner_idx <
             std::min(num_chunk_in_buffer, num_chunk - chunk_idx);
             rf_inner_idx++) {
          Limb::Ptr opnd = precomputed.at(x_idx + x_inner_idx);
          addr_vec[4] = opnd->GetRowAddr(chunk_idx + rf_inner_idx);
          addr_vec[5] = opnd->GetColAddr(chunk_idx + rf_inner_idx);
          pim_request.AddCommand(
              PIMCommand(PIMCommandType::kDRAM2RF, PIMOperandType::kPrecomputed,
                         addr_vec, &pim_request, dramreq_type));
        }
      }
      for (int x_inner_idx = 0;
           x_inner_idx < std::min(max_num_x_in_buffer, num_x - x_idx);
           x_inner_idx++) {
        for (int rf_inner_idx = 0;
             rf_inner_idx <
             std::min(num_chunk_in_buffer, num_chunk - chunk_idx);
             rf_inner_idx++) {
          Limb::Ptr opnd_b = src.at((x_idx + x_inner_idx) * 2);

          addr_vec[4] = opnd_b->GetRowAddr(chunk_idx + rf_inner_idx);
          addr_vec[5] = opnd_b->GetColAddr(chunk_idx + rf_inner_idx);
          pim_request.AddCommand(PIMCommand(PIMCommandType::kMAC,
                                            PIMOperandType::kSrc, addr_vec,
                                            &pim_request, dramreq_type));
        }
      }

      for (int x_inner_idx = 0;
           x_inner_idx < std::min(max_num_x_in_buffer, num_x - x_idx);
           x_inner_idx++) {
        for (int rf_inner_idx = 0;
             rf_inner_idx <
             std::min(num_chunk_in_buffer, num_chunk - chunk_idx);
             rf_inner_idx++) {
          Limb::Ptr opnd_a = src.at((x_idx + x_inner_idx) * 2 + 1);

          addr_vec[4] = opnd_a->GetRowAddr(chunk_idx + rf_inner_idx);
          addr_vec[5] = opnd_a->GetColAddr(chunk_idx + rf_inner_idx);
          pim_request.AddCommand(PIMCommand(PIMCommandType::kMAC,
                                            PIMOperandType::kSrc, addr_vec,
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