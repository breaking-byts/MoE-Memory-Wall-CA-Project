#include <algorithm>
#include <iostream>

#include "dram/dram_interface.h"
#include "dram/memory_object.h"
#include "dram/mmap_controller.h"
#include "dram/power.h"
#include "hardware/device.h"
#include "hardware/stat.h"
#include "module/tensor.h"

using namespace llm_system;

int main(int argc, char *argv[]) {
  std::vector<std::string> arg;
  for (int i = 1; i < argc; i++) {
    arg.push_back(argv[i]);
  }
  std::string output_path = arg[0];
  // std::string output_path = "test";

  // Area overhead //
  area_um2 dram_die = 106.92 * 1000.0 * 1000.0;
  area_um2 logic_die = 106.92 * 1000.0 * 1000.0;
  // area_um2 logic_die = 121.00 * 1000.0 * 1000.0;

  double dram_scale = 46;

  energy_nJ mac_energy = 0.45 / 1000;  // per operation
  area_um2 fp16_mac_logic = 139.2536;
  area_um2 fp16_mac_dram = 133.6257 * dram_scale;

  double pitch = 22;  // um
  area_um2 tsv_area = pitch * pitch * 4096 * 4;

  area_um2 scratch_pad_logic = 702.706;
  area_um2 scratch_pad_dram = 702.706 * dram_scale;

  area_um2 controller_logic = 613.0;
  area_um2 controller_dram = 613.0 * dram_scale;

  area_um2 compute_area_per_pe_logic = fp16_mac_logic * 16;
  area_um2 compute_area_per_pe_dram = fp16_mac_dram * 16;

  int num_pseudo_channel = 32;
  int num_rank = 2;
  int num_bankgroup = 4;
  int num_bank = 4;

  // Area overhead end//

  std::vector<energy_nJ> energy_list;
  std::vector<time_ns> delay_list;
  std::vector<area_um2> area_list;

  SystemConfig config;
  int m = 16384;
  int k = 4096;

  MemoryConfig memory_config(1);

  Device::Ptr device = Device::Create(config, 0, nullptr);
  device->mmap_controller = MMapController::Create(memory_config);
  Tensor::Ptr tensor = Tensor::Create("dummy", {m, k}, "act", device, device->model_config.precision_byte);

  std::vector<int> op_b = {1, 2, 4, 8, 16, 32};

  DRAMInterface::Ptr interface = DRAMInterface::Create("./dram_config.yaml", 1);

  // ALL-BANK PIM, X16 memory bandwidth //
  energy_nJ energy = 0;
  interface->setPIMHWConfig(ProcessorType::PIM, 16);

  DRAMRequest::Ptr dram_request = DRAMRequest::Create(DRAMRequestType::kGEMV);
  dram_request->AddOperand(tensor->getMemoryObject(), PIMOperandType::kSrc);

  double start_time = interface->time;
  std::list<DRAMRequest::Ptr> request;
  request.push_back(dram_request);
  interface->HandleRequest(request, 0);
  ExecStatus exec_status = interface->getExecStatus();

  energy += exec_status.act_count * pimEnergy.kACT_energy_j_;
  energy += exec_status.read_count * pimEnergy.kREAD_energy_j_;
  energy += exec_status.write_count * pimEnergy.kWRITE_energy_j_;
  energy += exec_status.all_act_count * pimEnergy.kALL_ACT_energy_j_;
  energy += exec_status.all_read_count * pimEnergy.kALL_READ_energy_j_;
  energy += exec_status.all_write_count * pimEnergy.kALL_WRITE_energy_j_;
  energy *= memory_config.num_channel;

  time_ns time = exec_status.memory_duration;

  for (int n : op_b) {
    energy_list.push_back(energy + 1.0 * m * k * n * mac_energy);
    delay_list.push_back(time * n);
    area_list.push_back(
        (dram_die +
         ((compute_area_per_pe_dram + controller_dram + scratch_pad_dram * n) *
          memory_config.num_bankgroup * memory_config.num_bank *
          memory_config.num_channel / 4)) /
        dram_die);
  }

  // ALL-BANKGroup PIM, X4 memory bandwidth, Op/b8 //
  energy = 0;
  interface->setPIMHWConfig(ProcessorType::PIM, 4);

  dram_request = DRAMRequest::Create(DRAMRequestType::kGEMV);
  dram_request->AddOperand(tensor->getMemoryObject(), PIMOperandType::kSrc);

  start_time = interface->time;
  request.resize(0);
  request.push_back(dram_request);
  interface->HandleRequest(request, 0);
  exec_status = interface->getExecStatus();

  energy += exec_status.act_count * pimBankgroupEnergy.kACT_energy_j_;
  energy += exec_status.read_count * pimBankgroupEnergy.kREAD_energy_j_;
  energy += exec_status.write_count * pimBankgroupEnergy.kWRITE_energy_j_;
  energy += exec_status.all_act_count * pimBankgroupEnergy.kALL_ACT_energy_j_;
  energy += exec_status.all_read_count * pimBankgroupEnergy.kALL_READ_energy_j_;
  energy +=
      exec_status.all_write_count * pimBankgroupEnergy.kALL_WRITE_energy_j_;
  energy *= memory_config.num_channel;

  time = exec_status.memory_duration;

  for (int n : op_b) {
    energy_list.push_back(energy + 1.0 * m * k * n * mac_energy);
    delay_list.push_back(time * std::max(1, n / 8));
    area_list.push_back((dram_die + (compute_area_per_pe_dram * 8 +
                                     controller_dram + scratch_pad_dram * n) *
                                        memory_config.num_bankgroup *
                                        memory_config.num_channel / 4) /
                        dram_die);
  }

  // LOGIC, X4 memory bandwidth //
  energy = 0;
  interface->setPIMHWConfig(ProcessorType::LOGIC, 4);

  dram_request = DRAMRequest::Create(DRAMRequestType::kGEMV);
  dram_request->AddOperand(tensor->getMemoryObject(), PIMOperandType::kSrc);

  start_time = interface->time;
  request.resize(0);
  request.push_back(dram_request);
  interface->HandleRequest(request, 0);
  exec_status = interface->getExecStatus();

  energy += exec_status.act_count * logicEnergy.kACT_energy_j_;
  energy += exec_status.read_count * logicEnergy.kREAD_energy_j_;
  energy += exec_status.write_count * logicEnergy.kWRITE_energy_j_;
  energy += exec_status.all_act_count * logicEnergy.kALL_ACT_energy_j_;
  energy += exec_status.all_read_count * logicEnergy.kALL_READ_energy_j_;
  energy += exec_status.all_write_count * logicEnergy.kALL_WRITE_energy_j_;
  energy *= memory_config.num_channel;

  time = exec_status.memory_duration;

  for (int n : op_b) {
    energy_list.push_back(energy + 1.0 * m * k * n * mac_energy);
    delay_list.push_back(time * std::max(1, n / 8));
    area_list.push_back((logic_die + tsv_area +
                         (compute_area_per_pe_logic * 8 + controller_logic +
                          scratch_pad_logic * n) *
                             4 * memory_config.num_channel) /
                        logic_die);
  }

  std::ofstream csv;
  std::string file_name = "unit_test.csv";

  csv.open(output_path + "/" + file_name);
  csv << "device,op_b,energy,delay,area" << std::endl;

  std::vector<std::string> device_list;
  device_list = {"PIM", "BG", "LOGIC"};
  int i = 0;
  for (auto device : device_list) {
    for (auto opb : op_b) {
      csv << device << "," << std::to_string(opb) << "," << energy_list[i]
          << "," << delay_list[i] << "," << area_list[i] << std::endl;
      i++;
    }
  }

  csv.close();

  return 0;
}
