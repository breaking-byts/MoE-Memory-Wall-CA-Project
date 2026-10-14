#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <iostream>

#include "hardware/stat.h"
#include "model/model.h"
#include "model/util.h"
#include "module/layer.h"
#include "module/module_graph.h"

using namespace llm_system;

int main(int argc, char *argv[]) {
  YAML::Node config;
  if(argc > 1){
    std::string config_path = argv[1];
    config = YAML::LoadFile(config_path);
  }
  else{
    config = YAML::LoadFile("config.yaml");
  }

  std::string model_name = config["model"]["model_name"].as<std::string>();
  std::string processor_type =
      config["system"]["processor_type"].as<std::string>();

  int num_node = config["system"]["num_node"].as<int>();
  int num_device = config["system"]["num_device"].as<int>();

  std::string data_name = config["simulation"]["data"].as<std::string>();
  int input_len = config["simulation"]["input_len"].as<int>();
  int output_len = config["simulation"]["output_len"].as<int>();
  int iter = config["simulation"]["iter"].as<int>();

  int max_batch_size = config["serving"]["max_batch_size"].as<int>();

  int max_process_token = config["serving"]["max_process_token"].as<int>();
  std::string output_path = config["log"]["output_directory"].as<std::string>();

  SystemConfig system_config;
  if(config["system"]["gpu_gen"].as<std::string>() == "A100"){
    system_config = A100;
  }
  else if (config["system"]["gpu_gen"].as<std::string>() == "H100"){
    system_config = H100;
  }
  else if (config["system"]["gpu_gen"].as<std::string>() == "B100"){
    system_config = B100;
  }
  else if (config["system"]["gpu_gen"].as<std::string>() == "B200"){
    system_config = B200;
  }
  else{
    fail("No GPU generation information");
  }

  // inter-GPU NVLink Config // 
  system_config.device_ict_bandwidth = 450.0 * 1000 * 1000 * 1000; // B/s NVLink 4th Gen (DGX H100)
  system_config.device_ict_latency = 0.8 * 1000; // ns

  // InfiniBand Config // 
  system_config.node_ict_bandwidth = 50.0 * 1000 * 1000 * 1000; // B/s Infiniband NDR
  system_config.node_ict_latency = 0.13 * 1000; // ns

  system_config.num_node = num_node;
  system_config.num_device = num_device;

  // default settings
  system_config.parallel_execution = false;
  system_config.hetero_subbatch = false;
  system_config.disagg_system = false;
  system_config.use_low_unit_moe_only = false;
  system_config.use_ramulator = false;
  system_config.use_flash_mla = false;
  system_config.use_flash_attention = false;
  system_config.reuse_kv_cache = false;
  system_config.kv_cache_reuse_rate = 0.0;
  
  // prefill mode or decode mode
  system_config.prefill_mode =
      config["system"]["optimization"]["prefill_mode"].as<bool>();
  system_config.decode_mode =
      config["system"]["optimization"]["decode_mode"].as<bool>();
  assertTrue((system_config.prefill_mode == false) || (system_config.decode_mode == false), 
            "prefill mode and decode mode is incompatible");

  assertTrue((system_config.parallel_execution == false) || (system_config.use_low_unit_moe_only == false),
            "parallel_execution & use_low_unit_moe_only are not compatible");
            
  if(system_config.prefill_mode){
    std::cout << "[Prefill Mode] Output Length is modified into 1" << std::endl;
  }

  if(system_config.decode_mode){
    std::cout << "[Decode Mode] Current Length of sequences is modified into input_len" << std::endl;
  }

  //// In this branch, we use only GPU-related code, neither LOGIC nor PIM
  if (!processor_type.compare("GPU")) {
    system_config.processor_type = {ProcessorType::GPU};
    system_config.high_processor_type = ProcessorType::GPU;
    system_config.low_processor_type = ProcessorType::GPU;
  } //else if (!processor_type.compare("LOGIC")) {
  //  system_config.processor_type = {ProcessorType::LOGIC};
  //  system_config.high_processor_type = ProcessorType::LOGIC;
  //  system_config.low_processor_type = ProcessorType::LOGIC;
  //} else if (!processor_type.compare("GPU+LOGIC")) {
  //  system_config.processor_type = {ProcessorType::GPU, ProcessorType::LOGIC};
  //  system_config.high_processor_type = ProcessorType::GPU;
  //  system_config.low_processor_type = ProcessorType::LOGIC;
  //} else if (!processor_type.compare("GPU+PIM")) {
  //  system_config.processor_type = {ProcessorType::GPU, ProcessorType::PIM};
  //  system_config.high_processor_type = ProcessorType::GPU;
  //  system_config.low_processor_type = ProcessorType::PIM;
  //}

  std::string expert_file_path;

  ModelConfig model_config;

  if (!model_name.compare("mixtral")) {
    model_config = mixtral;
  } else if (!model_name.compare("openMoE")) {
    model_config = openMoE;
  } else if (!model_name.compare("llama7bMoE")) {
    model_config = llama7bMoE;
  } else if (!model_name.compare("llama3_70B")) {
    model_config = llama3_70B;
  } else if (!model_name.compare("llama3_405B")) {
    model_config = llama3_405B;
  } else if (!model_name.compare("llama4_maverick")) {
    model_config = llama4_maverick;
  } else if (!model_name.compare("grok1")) {
    model_config = grok1;
  } else if (!model_name.compare("deepseekR1")) {
    model_config = deepseekR1;
  } else {
    fail("No model configuration of " + model_name);
  }

  model_config.e_tp_dg =
      config["system"]["distribution"]["expert_tensor_degree"].as<int>();
  model_config.ne_tp_dg =
      config["system"]["distribution"]["none_expert_tensor_degree"].as<int>();
  
  model_config.compressed_kv =
      config["system"]["optimization"]["compressed_kv"].as<bool>();
  model_config.use_absorb =
      config["system"]["optimization"]["use_absorb"].as<bool>();
  
  model_config.precision_byte = config["simulation"]["precision_byte"].as<int>();
  if(model_config.precision_byte == 1){ // if FP8 or INT8 
    system_config.compute_peak_flops *= 2; // system_config has FP16 peak FLOPS information
  }

  system_config.exit_out_of_memory = false;
  system_config.mem_cap_limit = false;

  model_config.dataset = data_name;

  if (!data_name.compare("synthesis")) {
    expert_file_path = "none";
    model_config.input_len = input_len;
    model_config.output_len = output_len;
  } else {
    expert_file_path =
        "../expert_data/experts_" + model_name + "_" + data_name + ".csv";
  }

  if((system_config.decode_mode == true) && (model_config.output_len <= 1)){
    fail("[Decode Mode] Output length must be larger than 1");
  }

  // long max_batch_size = 128;
  if (max_process_token == 0) {
    // max_process_token = 8192 * 16;
    max_process_token = 65536 * 8;
  }
  Scheduler::Ptr scheduler =
      Scheduler::Create(system_config, model_config, expert_file_path,
                        max_batch_size, 8192, max_process_token);

  Cluster::Ptr cluster = Cluster::Create(system_config, scheduler);

  Model model(model_config, cluster, scheduler);

  cluster->set_dependency();

  std::cout << "-----------------------------------" << std::endl;
  std::cout << "-------------start-----------------" << std::endl;
  std::cout << "-----------------------------------" << std::endl;

  std::vector<Stat> stat_list;
  int total_iter = iter;

  int ne_tp_dg = model_config.ne_tp_dg;
  int ne_dp_dg = system_config.num_device * num_node / ne_tp_dg;

  int precision_bytes = model_config.precision_byte;

  std::string offloading;
  if(config["system"]["offload"]["offload_expert_to_cpu_memory"].as<bool>()){
    offloading="CPU_memory";
  } else if(config["system"]["offload"]["offload_expert_to_ssd"].as<bool>()){
    offloading="SSD";
  } else{
    offloading="baseline"; // no offload
  }

  std::string collect;
  if(config["system"]["offload"]["collects_gpu_compute"].as<bool>()){
    collect="GPU_compute";
  } else{
    collect="offloaded_weight_copy";
  }

  int nvlink_gen = config["system"]["offload"]["offload_bandwidth"].as<int>();

  std::string file_name;
  file_name = output_path + "/" + model_name + "_" + data_name +
                        "_N" + std::to_string(num_node) + "_D" + 
                        std::to_string(num_device) + "_TP" +
                        std::to_string(ne_tp_dg) + "_DP" +
                        std::to_string(ne_dp_dg) + "_batchsize" +
                        std::to_string(max_batch_size) + "_" + offloading + 
                        "_NVLink" + std::to_string(nvlink_gen) + "_" + collect + ".csv";

  
  scheduler->getActualArrivalTime(total_iter);
  stat_list = cluster->runIteration(total_iter, file_name);

  //TopModuleGraph::Ptr top0 = cluster->get_device(0)->top_module_graph;
  //top0->print_timeboard();

  return 0;
}
