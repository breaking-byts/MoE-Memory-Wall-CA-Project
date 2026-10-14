#include <algorithm>

#include "hardware/stat.h"
#include "model/model.h"
#include "model/util.h"
#include "module/layer.h"
#include "module/module_graph.h"

using namespace llm_system;

int main() {
  SystemConfig system_config;
  system_config.num_node = 1;
  system_config.num_device = 4;
  // system_config.processor_type = {ProcessorType::GPU};
  system_config.processor_type = {ProcessorType::GPU, ProcessorType::LOGIC};
  system_config.high_processor_type = ProcessorType::GPU;
  system_config.low_processor_type = ProcessorType::LOGIC;

  system_config.parallel_execution = false;

  system_config.disagg_system = true;

  // system_config.use_ramulator = false;
  system_config.use_ramulator = true;

  ModelConfig& model_config = mixtral;
  model_config.ne_tp_dg = system_config.num_device;
  model_config.e_tp_dg = system_config.num_device;
  model_config.num_layers = 1;

  // model_config.expert_freq = 1;
  model_config.input_len = 2048;
  model_config.output_len = 512;

  long max_batch_size = 16;

  Scheduler::Ptr scheduler = Scheduler::Create(
      system_config, model_config, "none",
      max_batch_size, 4096);

  Cluster::Ptr cluster = Cluster::Create(system_config, scheduler);

  scheduler->fillSequenceQueue();
  scheduler->fillRunningQueue();
  
  Model model(model_config, cluster, scheduler);

  cluster->set_dependency();

  cluster->checkMemorySize();

  std::cout << "-----------------------------------" << std::endl;
  std::cout << "-------------start-----------------" << std::endl;
  std::cout << "-----------------------------------" << std::endl;

  std::vector<Stat> stat_list;
  int total_iter = 2;
  stat_list = cluster->runIteration(total_iter);

  total_iter = stat_list.size();
  int p50 = total_iter / 2;
  int p99 = total_iter * 0.99;
  int p99dot9 = total_iter * 0.999;

  std::vector<time_ns> time;
  for (auto& temp : stat_list) {
    if (temp.iter_info) {
      time.push_back(temp.latency);
    }
  }

  sort(time.begin(), time.end());
  // std::cout << "p50: " << time.at(p50) << ", p99: " << time.at(p99)
  //           << ", p99.9: " << time.at(p99dot9) << std::endl;

  TopModuleGraph::Ptr top0 = cluster->get_device(0)->top_module_graph;
  TopModuleGraph::Ptr top1 = cluster->get_device(1)->top_module_graph;
  TopModuleGraph::Ptr top2 = cluster->get_device(2)->top_module_graph;
  TopModuleGraph::Ptr top3 = cluster->get_device(3)->top_module_graph;

  top0->print_timeboard();
  top1->print_timeboard();
  top2->print_timeboard();
  top3->print_timeboard();

  cluster->getTotalEnergy();
  // top1->print_timeboard();

  return 0;
}
