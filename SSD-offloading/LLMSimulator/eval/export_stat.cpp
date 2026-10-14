#include <algorithm>
#include <iostream>

#include "hardware/stat.h"
#include "model/model.h"
#include "model/util.h"
#include "module/layer.h"
#include "module/module_graph.h"

using namespace llm_system;

int main(int argc, char *argv[]) {
  std::vector<std::string> arg;
  for (int i = 1; i < argc; i++) {
    arg.push_back(argv[i]);
  }
  // 0: data_name,  1: processor_type, 2: device_num;

  std::string model_name = arg[0];
  std::string data_name = arg[1];
  std::string processor_type = arg[2];
  int num_device = std::stoi(arg[3]);
  std::string output_path = arg[4];
  // int max_batch_size = 128;
  //  int iter = std::stoi(arg[6]);
  int max_batch_size = std::stoi(arg[5]);
  int input_len = std::stoi(arg[6]);
  int output_len = std::stoi(arg[7]);
  int max_process_token = std::stoi(arg[8]);
  int request_per_second = std::stoi(arg[9]);
  int iter = std::stoi(arg[10]);
  int num_node = std::stoi(arg[11]);
  int e_tp_dg = std::stoi(arg[12]);
  int parallel_execution = std::stoi(arg[13]);
  int hetero_subbatch = std::stoi(arg[14]); 
  int is_prefill = std::stoi(arg[15]); 


  SystemConfig system_config;
  system_config.num_node = num_node;
  system_config.num_device = num_device;

  // system_config.processor_type = {ProcessorType::GPU};
  system_config.high_processor_type = ProcessorType::GPU;
  system_config.low_processor_type = ProcessorType::LOGIC;

  system_config.parallel_execution = false;
  system_config.hetero_subbatch = false;
  system_config.disagg_system = false;


  system_config.use_ramulator = false;

  system_config.use_flash_mla = false;
  system_config.use_flash_attention = false;

  system_config.reuse_kv_cache = false;
  system_config.kv_cache_reuse_rate = 0.0;

  if(is_prefill){
    system_config.prefill_mode = true;
    system_config.decode_mode = false;
  }
  else{
    system_config.prefill_mode = false;
    system_config.decode_mode = true;
  }

  assertTrue((system_config.prefill_mode == false) || (system_config.decode_mode == false), 
            "prefill mode and decode mode is incompatible");
  if(system_config.prefill_mode){
    std::cout << "[Prefill Mode] Output Length is modified into 1" << std::endl;
  }

  if(system_config.decode_mode){
    std::cout << "[Decode Mode] Current Length of sequences is modified into input_len" << std::endl;
  }

  int split = 0;
  if (!system_config.disagg_system) {
    split = 1;
  }

  if (!processor_type.compare("GPU")) {
    system_config.processor_type = {ProcessorType::GPU};
  } else if (!processor_type.compare("LOGIC")) {
    system_config.processor_type = {ProcessorType::LOGIC};
  } else if (!processor_type.compare("GPU+LOGIC")) {
    system_config.processor_type = {ProcessorType::GPU, ProcessorType::LOGIC};
    if (parallel_execution) {
      system_config.parallel_execution = true;
    } else {
      system_config.parallel_execution = false;
    }
    if (hetero_subbatch) {
      system_config.parallel_execution = true;
      system_config.hetero_subbatch = true;
    }
  } else if (!processor_type.compare("GPU+PIM")) {
    system_config.processor_type = {ProcessorType::GPU, ProcessorType::PIM};
    system_config.parallel_execution = false;
  } else {
    fail("Not supported device type");
  }

  if (request_per_second == 0) {
    system_config.use_inject_rate = false;
  } else {
    system_config.use_inject_rate = true;
    system_config.request_per_second = request_per_second;
  }

  std::string expert_file_path;

  ModelConfig model_config;

  if (!model_name.compare("mixtral")) {
    model_config = mixtral;
  } else if (!model_name.compare("openMoE")) {
    model_config = openMoE;
  } else if (!model_name.compare("llama7bMoE")) {
    model_config = llama7bMoE;
  } else if (!model_name.compare("grok1")) {
    model_config = grok1;
  } else if (!model_name.compare("glam")) {
    model_config = glam;
  } else if (!model_name.compare("deepseekR1")) {
    model_config = deepseekR1;
  } else {
    fail("Model is not supported");
  }

  model_config.ne_tp_dg = system_config.num_device;
  model_config.e_tp_dg = e_tp_dg;
  model_config.dataset = data_name;
  if(system_config.prefill_mode){
    model_config.use_absorb = false;
    model_config.compressed_kv = true;
    assertTrue(max_batch_size == 1, "For prefill mode, max batch size must be 1");
  }
  if(system_config.decode_mode){
    model_config.use_absorb = true;
    model_config.compressed_kv = true;
  }
  // model_config.max_seq_len = 512;

  if (!data_name.compare("synthesis")) {
    expert_file_path = "none";
    model_config.input_len = input_len;
    model_config.output_len = output_len;
  } else {
    expert_file_path =
        "../expert_data/experts_" + model_name + "_" + data_name + ".csv";
  }

  // long max_batch_size = 128;
  if (max_process_token == 0) {
    max_process_token = 8192 * 4;
  }
  Scheduler::Ptr scheduler =
      Scheduler::Create(system_config, model_config, expert_file_path,
                        max_batch_size, 8192 * 32, max_process_token);

  Cluster::Ptr cluster = Cluster::Create(system_config, scheduler);

  Model model(model_config, cluster, scheduler);

  bool out_of_memory = cluster->checkMemorySize();
  cluster->set_dependency();

  // std::vector<Stat> stat_list;
  int total_iter = iter;
  
  std::ofstream csv;
    std::string file_name;
  if(system_config.prefill_mode){
      file_name = output_path + "/" + model_name + "_" + data_name + "_" +
      std::to_string(input_len) + "_" + std::to_string(output_len) + "_" +
      processor_type + "_N" + std::to_string(num_node) + "_D" +
      std::to_string(num_device) + "_maxbatch" +
      std::to_string(max_batch_size) + "_maxprocess" +
      std::to_string(max_process_token) + "_injectionrate" +
      std::to_string(request_per_second) + "_iter" +
      std::to_string(total_iter) + "_e_tp" + std::to_string(e_tp_dg) +
      "_parallel_execution" + std::to_string(parallel_execution) + "_split" +
      std::to_string(split) + "_hetero_subbatch" +
      std::to_string(hetero_subbatch) + "_prefill.csv";

  }else if(system_config.decode_mode){
    file_name = output_path + "/" + model_name + "_" + data_name + "_" +
    std::to_string(input_len) + "_" + std::to_string(output_len) + "_" +
    processor_type + "_N" + std::to_string(num_node) + "_D" +
    std::to_string(num_device) + "_maxbatch" +
    std::to_string(max_batch_size) + "_maxprocess" +
    std::to_string(max_process_token) + "_injectionrate" +
    std::to_string(request_per_second) + "_iter" +
    std::to_string(total_iter) + "_e_tp" + std::to_string(e_tp_dg) +
    "_parallel_execution" + std::to_string(parallel_execution) + "_split" +
    std::to_string(split) + "_hetero_subbatch" +
    std::to_string(hetero_subbatch) + "_decode.csv";
  }
  else{
    file_name = output_path + "/" + model_name + "_" + data_name + "_" +
    std::to_string(input_len) + "_" + std::to_string(output_len) + "_" +
    processor_type + "_N" + std::to_string(num_node) + "_D" +
    std::to_string(num_device) + "_maxbatch" +
    std::to_string(max_batch_size) + "_maxprocess" +
    std::to_string(max_process_token) + "_injectionrate" +
    std::to_string(request_per_second) + "_iter" +
    std::to_string(total_iter) + "_e_tp" + std::to_string(e_tp_dg) +
    "_parallel_execution" + std::to_string(parallel_execution) + "_split" +
    std::to_string(split) + "_hetero_subbatch" +
    std::to_string(hetero_subbatch) + ".csv";
  }

  std::cout << "-----------------------------------" << std::endl;
  std::cout << "-------------start-----------------" << std::endl;
  std::cout << "-----------------------------------" << std::endl;

  scheduler->getActualArrivalTime(total_iter);
  cluster->runIteration(total_iter, file_name);

  // TopModuleGraph::Ptr top0 = cluster->get_device(0)->top_module_graph;
  // top0->print_timeboard();
  return 0;
}
