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
  int iter = std::stoi(arg[5]);

  SystemConfig system_config;
  system_config.num_node = 1;
  system_config.num_device = num_device;

  // system_config.processor_type = {ProcessorType::GPU};
  system_config.high_processor_type = ProcessorType::GPU;
  system_config.low_processor_type = ProcessorType::LOGIC;

  system_config.parallel_execution = false;
  system_config.communication_hiding = false;
  system_config.hetero_subbatch = false;

  system_config.disagg_system = true;

  system_config.use_ramulator = true;

  if (!processor_type.compare("GPU")) {
    system_config.processor_type = {ProcessorType::GPU};
  } else if (!processor_type.compare("LOGIC")) {
    system_config.processor_type = {ProcessorType::LOGIC};
  } else if (!processor_type.compare("GPU+LOGIC")) {
    system_config.processor_type = {ProcessorType::GPU, ProcessorType::LOGIC};
    system_config.parallel_execution = true;
  }

  std::string expert_file_path =
      "../expert_data/experts_" + model_name + "_" + data_name + ".csv";

  ModelConfig model_config;
  model_config.ne_tp_dg = system_config.num_device;
  model_config.e_tp_dg = system_config.num_device;

  model_config.num_layers = 32;
  model_config.expert_freq = 4;

  long max_batch_size = 128;

  Scheduler::Ptr scheduler = Scheduler::Create(
      system_config, model_config, expert_file_path, max_batch_size, 4096);

  Cluster::Ptr cluster = Cluster::Create(system_config, scheduler);

  Model model(model_config, cluster, scheduler);

  cluster->checkMemorySize();
  cluster->set_dependency();

  std::cout << "-----------------------------------" << std::endl;
  std::cout << "-------------start-----------------" << std::endl;
  std::cout << "-----------------------------------" << std::endl;

  std::vector<Stat> stat_list;
  int total_iter = iter;
  int p50 = total_iter / 2;
  int p99 = total_iter * 0.99;
  int p99dot9 = total_iter * 0.999;
  stat_list = cluster->runIteration(total_iter);

  TopModuleGraph::Ptr top0 = cluster->get_device(0)->top_module_graph;
  // TopModuleGraph::Ptr top1 = cluster->get_device(1)->top_module_graph;
  // top0->print_timeboard();
  // top1->print_timeboard();

  std::ofstream csv;
  std::string file_name = model_name + "_" + data_name + "_" + processor_type +
                          std::to_string(num_device) + "_" +
                          std::to_string(max_batch_size) + "_iter" +
                          std::to_string(total_iter) + ".csv";
  csv.open(output_path + "/" + file_name);
  csv << "latency,mixed,batchsize,numtoken,seqlen" << std::endl;
  for (auto temp : stat_list) {
    csv << std::to_string(temp.latency) << ","
        << std::to_string(temp.is_mixed) << ","
        << std::to_string(temp.batchsize) << ","
        << std::to_string(temp.process_token) << ","
        << std::to_string(temp.average_seq_len) << std::endl;
  }
  csv.close();

  return 0;
}
