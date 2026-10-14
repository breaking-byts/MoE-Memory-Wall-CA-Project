#pragma once

#include <yaml-cpp/yaml.h>
#include <string>
#include <vector>

#include "common/type.h"
#include "hardware/base.h"

namespace llm_system {

struct PIMHWConfig {
  ProcessorType type = ProcessorType::GPU;

  int bandwidth_x = 0;
};

class SystemConfig {

  public:
    // default initilizing with H100 config
    SystemConfig(std::string gpu_gen ="H100",
                 int num_node = 1, int num_device = 2, 
                 hw_metric node_ict_latency = 0.5 * 1000,
                 hw_metric node_ict_bandwidth = 400.0 * 1000 * 1000 * 1000,
                 hw_metric device_ict_latency = 3.0 * 1000,
                 hw_metric device_ict_bandwidth = 450.0 * 1000 * 1000 * 1000, 
                 hw_metric compute_peak_flops = 989.4 * 1000 * 1000 * 1000 * 1000,
                 hw_metric base_memory_bandwidth = 3.352 * 1000 * 1000 * 1000 * 1000,
                 hw_metric memory_capacity = 80.0 * 1024 * 1024 * 1024,\
                 int logic_x = 4,
                 hw_metric logic_op_b = 8,                 
                 int pim_x = 16,
                 hw_metric pim_op_b = 1,
                 std::vector<ProcessorType> processor_type = {},
                 bool parallel_execution = false,
                 bool hetero_subbatch = false,
                 ProcessorType high_processor_type = ProcessorType::GPU,
                 ProcessorType low_processor_type = ProcessorType::LOGIC,
                 bool communication_hiding = false,            
                 bool disagg_system = false,
                 bool use_low_unit_moe_only = false,
                 bool use_ramulator = false,
                 bool exit_out_of_memory = true,
                 bool mem_cap_limit = false,               
                 bool use_flash_mla = true,
                 bool use_flash_attention = true,
                 bool reuse_kv_cache = true,
                 hw_metric kv_cache_reuse_rate = 0.5,
                 bool prefill_mode = false,
                 bool decode_mode = false,
                 bool use_inject_rate = false,
                 int request_per_second = 10,
                 int num_cube = 5,
                 int num_logic_cube = 5
                )
      : gpu_gen(gpu_gen),
        num_node(num_node),
        num_device(num_device),
        node_ict_latency(node_ict_latency),
        node_ict_bandwidth(node_ict_bandwidth),
        device_ict_latency(device_ict_latency),
        device_ict_bandwidth(device_ict_bandwidth),
        compute_peak_flops(compute_peak_flops),
        base_memory_bandwidth(base_memory_bandwidth),
        memory_bandwidth(memory_bandwidth),
        memory_capacity(memory_capacity),
        logic_x(logic_x),
        logic_op_b(logic_op_b),
        pim_x(pim_x),
        pim_op_b(pim_op_b),
        processor_type(processor_type),
        parallel_execution(parallel_execution),
        hetero_subbatch(hetero_subbatch),
        high_processor_type(high_processor_type),
        low_processor_type(low_processor_type),
        communication_hiding(communication_hiding),
        disagg_system(disagg_system),
        use_low_unit_moe_only(use_low_unit_moe_only),
        use_ramulator(use_ramulator),
        exit_out_of_memory(exit_out_of_memory),
        mem_cap_limit(mem_cap_limit),
        use_flash_mla(use_flash_mla),
        use_flash_attention(use_flash_attention),
        reuse_kv_cache(reuse_kv_cache),
        kv_cache_reuse_rate(kv_cache_reuse_rate),
        prefill_mode(prefill_mode),
        decode_mode(decode_mode),
        use_inject_rate(use_inject_rate),
        request_per_second(request_per_second),
        num_cube(num_cube),
        num_logic_cube(num_logic_cube),
        l1_cache_size(256.0 * 1024),
        l2_cache_size(50.0 * 1024 * 1024) {
          parseOffloadConfig();

          memory_bandwidth = base_memory_bandwidth - offload_bandwidth;

          logic_memory_bandwidth = memory_bandwidth * logic_x;
          pim_memory_bandwidth = memory_bandwidth * pim_x;
        }

  SystemConfig& operator=(const SystemConfig& rhs) = default;

  std::string gpu_gen;

  // Device number
  int num_node;
  int num_device;

  // Cluster specification
  hw_metric node_ict_latency;   // ns
  hw_metric node_ict_bandwidth; // B/s

  // Node specification
  hw_metric device_ict_latency;    // ns, 
  hw_metric device_ict_bandwidth;  // B/s

  // Device specification
  hw_metric l1_cache_size;
  hw_metric l2_cache_size;
  hw_metric compute_peak_flops;
  hw_metric memory_bandwidth; // HBM read/write bandwidth - SSD read bandwidth
  hw_metric base_memory_bandwidth; // HBM read/write bandwidth
  hw_metric memory_capacity;
  hw_metric offload_bandwidth;

  // not used
  int logic_x;
  hw_metric logic_memory_bandwidth = memory_bandwidth * logic_x;
  hw_metric logic_op_b;

  // not used
  int pim_x;
  hw_metric pim_memory_bandwidth = memory_bandwidth * pim_x;
  hw_metric pim_op_b;

  std::vector<ProcessorType> processor_type = {};

  bool parallel_execution = false;
  bool hetero_subbatch = false;
  ProcessorType high_processor_type = ProcessorType::GPU;
  ProcessorType low_processor_type = ProcessorType::LOGIC;

  bool communication_hiding = false;

  bool disagg_system = false;
  bool use_low_unit_moe_only = false;
  bool use_ramulator = false;
  
  bool exit_out_of_memory = false;
  bool mem_cap_limit = false;

  bool use_flash_mla = false; 
  bool use_flash_attention = false; 
  bool reuse_kv_cache = false;
  hw_metric kv_cache_reuse_rate; 

  bool prefill_mode = false; 
  bool decode_mode = false;

  bool use_inject_rate = false;  // injection random number of sequence
  int request_per_second;

  int num_cube; //8: for HBM3E (B100), 5 for HBM3 (H100)
  int num_logic_cube;

  private:
    void parseOffloadConfig() {
      YAML::Node config = YAML::LoadFile("config.yaml");

      bool offload_cpu = config["system"]["offload"]["offload_expert_to_cpu_memory"].as<bool>();
      bool offload_ssd = config["system"]["offload"]["offload_expert_to_ssd"].as<bool>();

      if (offload_cpu || offload_ssd) {
        int gen = config["system"]["offload"]["offload_bandwidth"].as<int>();
        if (gen == 4) {
          offload_bandwidth = 450.0 * 1000 * 1000 * 1000;  // NVLink 4.0
        } else if (gen == 5) {
          offload_bandwidth = 900.0 * 1000 * 1000 * 1000;  // NVLink 5.0
        }
      }
      else {
        offload_bandwidth = 0.0;
      }
    }  
};


static SystemConfig A100 = SystemConfig(
                 "A100",                            // gpu gen
                 1,                                 // num_node 
                 2,                                 // num_device
                 130.0,                             // node_ict_latency, connectx-7 https://www.fs.com/products/161048.html?attribute=106827&id=3941024
                 50.0 * 1000 * 1000 * 1000,         // node_ict_bandwidth
                 3.0 * 1000,                        // device_ict_latency, nvlink 3
                 150.0 * 1000 * 1000 * 1000,        // device_ict_bandwidth
                 312.0 * 1000 * 1000 * 1000 * 1000, // compute_peak_flops, FP16
                 2.039 * 1000 * 1000 * 1000 * 1000, // base_memory_bandwidth
                 80.0 * 1024 * 1024 * 1024,         // memory_capacity 
                 4,                                 // logic_x 
                 8,                                 // logic_op_b                 
                 16,                                // pim_x
                 1,                                 // pim_op_b
                 {},                                // processor_type
                 false,                             // parallel_execution
                 false,                             // hetero_subbatch
                 ProcessorType::GPU,                // high_processor_type
                 ProcessorType::LOGIC,              // low_processor_type
                 false,                             // communication_hiding
                 false,                             // disagg_system 
                 false,                             // use_low_unit_moe_only
                 false,                             // use_ramulator
                 true,                              // exit_out_of_memory
                 false,                             // mem_cap_limit
                 true,                              // use_flash_mla
                 true,                              // use_flash_attention
                 false,                             // reuse_kv_cache
                 0.0,                               // kv_cache_reuse_rate
                 false,                             // prefill_mode
                 false,                             // decode_mode
                 false,                             // use_inject_rate
                 10,                                // request_per_second
                 5,                                 // num_cube
                 5                                  // int num_logic_cube
                 );

static SystemConfig H100 = SystemConfig(
                 "H100",                            // gpu gen
                 1,                                 // num_node 
                 2,                                 // num_device
                 130.0,                             // node_ict_latency, connectx-7
                 50.0 * 1000 * 1000 * 1000,         // node_ict_bandwidth
                 0.8 * 1000,                        // device_ict_latency
                 450.0 * 1000 * 1000 * 1000,        // device_ict_bandwidth, nvlink 4
                 989.4 * 1000 * 1000 * 1000 * 1000, // compute_peak_flops, FP16
                 3.352 * 1000 * 1000 * 1000 * 1000, // base_memory_bandwidth
                 80.0 * 1024 * 1024 * 1024,         // memory_capacity 
                 4,                                 // logic_x 
                 8,                                 // logic_op_b                 
                 16,                                // pim_x
                 1,                                 // pim_op_b
                 {},                                // processor_type
                 false,                             // parallel_execution
                 false,                             // hetero_subbatch
                 ProcessorType::GPU,                // high_processor_type
                 ProcessorType::LOGIC,              // low_processor_type
                 false,                             // communication_hiding
                 false,                             // disagg_system
                 false,                             // use_low_unit_moe_only 
                 false,                             // use_ramulator
                 true,                              // exit_out_of_memory
                 false,                             // mem_cap_limit
                 true,                              // use_flash_mla
                 true,                              // use_flash_attention
                 false,                             // reuse_kv_cache
                 0.0,                               // kv_cache_reuse_rate
                 false,                             // prefill_mode
                 false,                             // decode_mode
                 false,                             // use_inject_rate
                 10,                                // request_per_second
                 5,                                 // num_cube
                 5                                  // int num_logic_cube
                 );

static SystemConfig B100 = SystemConfig(
                  "B100",                            // gpu gen
                  1,                                 // num_node 
                  2,                                 // num_device
                  130.0,                             // node_ict_latency, connectx-7 
                  50.0 * 1000 * 1000 * 1000,         // node_ict_bandwidth
                  0.8 * 1000,                        // device_ict_latency, nvlink 5.0
                  900.0 * 1000 * 1000 * 1000,        // device_ict_bandwidth
                  1750.0 * 1000 * 1000 * 1000 * 1000,// compute_peak_flops, FP16
                  8.000 * 1000 * 1000 * 1000 * 1000, // base_memory_bandwidth
                  192.0 * 1024 * 1024 * 1024,        // memory_capacity 
                  4,                                 // logic_x 
                  8,                                 // logic_op_b                 
                  16,                                // pim_x
                  1,                                 // pim_op_b
                  {},                                // processor_type
                  false,                             // parallel_execution
                  false,                             // hetero_subbatch
                  ProcessorType::GPU,                // high_processor_type
                  ProcessorType::LOGIC,              // low_processor_type
                  false,                             // communication_hiding
                  false,                             // disagg_system
                  false,                             // use_low_unit_moe_only 
                  false,                             // use_ramulator
                  true,                              // exit_out_of_memory
                  false,                             // mem_cap_limit
                  true,                              // use_flash_mla
                  true,                              // use_flash_attention
                  false,                             // reuse_kv_cache
                  0.0,                               // kv_cache_reuse_rate
                  false,                             // prefill_mode
                  false,                             // decode_mode
                  false,                             // use_inject_rate
                  10,                                // request_per_second
                  8,                                 // num_cube
                  8                                  // int num_logic_cube
                  );
                  
static SystemConfig B200 = SystemConfig(
                 "B200",                            // gpu gen
                 1,                                 // num_node 
                 2,                                 // num_device
                 130.0,                             // node_ict_latency, connectx-7 
                 50.0 * 1000 * 1000 * 1000,         // node_ict_bandwidth
                 0.8 * 1000,                        // device_ict_latency, nvlink 5.0
                 900.0 * 1000 * 1000 * 1000,        // device_ict_bandwidth
                 2250.0 * 1000 * 1000 * 1000 * 1000,// compute_peak_flops, FP16
                 8.000 * 1000 * 1000 * 1000 * 1000, // base_memory_bandwidth
                 192.0 * 1024 * 1024 * 1024,        // memory_capacity 
                 4,                                 // logic_x 
                 8,                                 // logic_op_b                 
                 16,                                // pim_x
                 1,                                 // pim_op_b
                 {},                                // processor_type
                 false,                             // parallel_execution
                 false,                             // hetero_subbatch
                 ProcessorType::GPU,                // high_processor_type
                 ProcessorType::LOGIC,              // low_processor_type
                 false,                             // communication_hiding
                 false,                             // disagg_system
                 false,                             // use_low_unit_moe_only 
                 false,                             // use_ramulator
                 true,                              // exit_out_of_memory
                 false,                             // mem_cap_limit
                 true,                              // use_flash_mla
                 true,                              // use_flash_attention
                 false,                             // reuse_kv_cache
                 0.0,                               // kv_cache_reuse_rate
                 false,                             // prefill_mode
                 false,                             // decode_mode
                 false,                             // use_inject_rate
                 10,                                // request_per_second
                 8,                                 // num_cube
                 8                                  // int num_logic_cube
                 );

}  // namespace llm_system