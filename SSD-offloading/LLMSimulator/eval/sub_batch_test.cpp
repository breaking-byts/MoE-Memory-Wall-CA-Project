#include <algorithm>

#include "model/model.h"
#include "model/util.h"
#include "module/layer.h"
#include "module/module_graph.h"
#define MAX(a, b) a > b ? a : b

using namespace llm_system;

int main() {
  SystemConfig system_config;
  system_config.num_node = 1;
  system_config.num_device = 2;
  system_config.processor_type = {ProcessorType::GPU, ProcessorType::LOGIC};
  system_config.hetero_subbatch = true;
  system_config.use_ramulator = false;

  ModelConfig model_config;
  model_config.ne_tp_dg = 2;
  model_config.e_tp_dg = 1;
  model_config.num_layers = 2;
  model_config.expert_freq = 1;

  long max_batch_size = 32;

  Scheduler::Ptr scheduler1 = Scheduler::Create(
      system_config, model_config, "../expert_data/experts_mixtral_GSM.csv",
      max_batch_size / 2, 4096);
  Scheduler::Ptr scheduler2 = Scheduler::Create(
      system_config, model_config, "../expert_data/experts_mixtral_GSM.csv",
      max_batch_size / 2, 4096);

  Cluster::Ptr cluster1 = Cluster::Create(system_config, scheduler1);
  Cluster::Ptr cluster2 = Cluster::Create(system_config, scheduler2);

  Model model1(model_config, cluster1, scheduler1);
  Model model2(model_config, cluster2, scheduler2);

  cluster1->checkMemorySize();
  cluster2->checkMemorySize();
  
  cluster1->set_dependency();
  cluster2->set_dependency();

  std::cout << "-----------------------------------" << std::endl;
  std::cout << "-------------start-----------------" << std::endl;
  std::cout << "-----------------------------------" << std::endl;

  int total_iter = 60;
  int p50 = total_iter / 2;
  int p99 = total_iter * 0.99;
  int p99dot9 = total_iter * 0.999;
  // time = cluster->runIteration(total_iter);
  
  scheduler1->fillSequenceQueue();
  scheduler1->fillRunningQueue();

  scheduler2->fillSequenceQueue();
  scheduler2->fillRunningQueue();

  // hitting
  // scheduler1->hittingQueue(5000);
  // scheduler2->hittingQueue(5000);
  
  time_ns total_time = 0;

  std::vector<time_ns> time_list;

  for (int i = 0; i < total_iter; i++) {
    auto metadata1 = scheduler1->setMetadata();
    auto metadata2 = scheduler2->setMetadata();
    cluster1->run(metadata1);
    cluster2->run(metadata2);

    TimeBoard& timeboard1 =
        cluster1->get_device(0)->top_module_graph->timeboard;
    TimeBoard& timeboard2 =
        cluster2->get_device(0)->top_module_graph->timeboard;

    std::vector<TimeStamp*> QKV_gen_1; // GPU
    std::vector<TimeStamp*> AttnSum_1; // GPU
    std::vector<TimeStamp*> AttnGen_1; // PIM or Logic
    std::vector<TimeStamp*> O_proj_1; // GPU
    std::vector<TimeStamp*> FFN_1; // GPU
    std::vector<TimeStamp*> ExpertFFN_1; // PIM or Logic

    std::vector<TimeStamp*> QKV_gen_2; // GPU
    std::vector<TimeStamp*> AttnSum_2; // GPU
    std::vector<TimeStamp*> AttnGen_2; // PIM or Logic
    std::vector<TimeStamp*> O_proj_2; // GPU
    std::vector<TimeStamp*> FFN_2; // GPU
    std::vector<TimeStamp*> ExpertFFN_2; // PIM or Logic

    timeboard1.find_stamp("attn_qkv_proj", QKV_gen_1);
    timeboard1.find_stamp("AttentionSum", AttnSum_1);
    timeboard1.find_stamp("AttentionGen", AttnGen_1);
    timeboard1.find_stamp("attn_o_proj", O_proj_1);
    timeboard1.find_stamp("expertFFN", ExpertFFN_1);
    timeboard1.find_stamp("feedforward", FFN_1);

    timeboard2.find_stamp("attn_qkv_proj", QKV_gen_2);
    timeboard2.find_stamp("AttentionSum", AttnSum_2);
    timeboard2.find_stamp("AttentionGen", AttnGen_2);
    timeboard2.find_stamp("attn_o_proj", O_proj_2);
    timeboard2.find_stamp("expertFFN", ExpertFFN_2);
    timeboard1.find_stamp("feedforward", FFN_2);

    time_ns stage_time = 0;
    int layer_idx_1 = 0;
    int layer_idx_2 = 0;
    
    int cur_1 = 1;
    int cur_2 = 0;

    time_ns pipe_1;
    time_ns pipe_2;

    int num_layer = model_config.num_layers;
    int expert_freq = model_config.expert_freq;
    stage_time += QKV_gen_1[0]->get_duration() + AttnSum_1[0]->get_duration();
    while(layer_idx_1 < num_layer || layer_idx_2 < num_layer){
      // first layer must be MoE layer
      // batch-1
      pipe_1 = 0;
      pipe_2 = 0;
      if(layer_idx_1 % expert_freq == 0 && layer_idx_1 < num_layer){ // is MoE Layer
        if(cur_1 == 0){
          pipe_1 = QKV_gen_1[layer_idx_1]->get_duration() + AttnSum_1[layer_idx_1]->get_duration();
        }
        else if(cur_1 == 1){
          pipe_1 = AttnGen_1[layer_idx_1]->get_duration();
        }
        else if(cur_1 == 2){
          pipe_1 = O_proj_1[layer_idx_1]->get_duration();
        }
        else {
          int expert_ffn_idx = layer_idx_1 / expert_freq;
          pipe_1 = ExpertFFN_1[expert_ffn_idx]->get_duration();
        }
      }
      else if (layer_idx_1 < num_layer){ // is Non-MoE layer
        if(cur_1 == 0){
          pipe_1 = QKV_gen_1[layer_idx_1]->get_duration() + AttnSum_1[layer_idx_1]->get_duration();
        }
        else if(cur_1 == 1){
          pipe_1 = AttnGen_1[layer_idx_1]->get_duration();
        }
        else {
          int ffn_idx = layer_idx_1 - (layer_idx_1 / expert_freq) - 1;
          if(layer_idx_1 != num_layer - 1){
            pipe_1 = O_proj_1[layer_idx_1]->get_duration()
                    + FFN_1[ffn_idx]->get_duration()
                    + QKV_gen_1[layer_idx_1 + 1]->get_duration() + AttnSum_1[layer_idx_1 + 1]->get_duration();
          }
          else{
            pipe_1 = O_proj_1[layer_idx_1]->get_duration()
                    + FFN_1[ffn_idx]->get_duration();
          }
          layer_idx_1++;
          cur_1 = 0;
        }
      }
      // batch-2
      if(layer_idx_2 % expert_freq == 0){ // is MoE Layer
        if(cur_2 == 0){
          pipe_2 = QKV_gen_2[layer_idx_2]->get_duration() + AttnSum_2[layer_idx_2]->get_duration();
        }
        else if(cur_2 == 1){
          pipe_2 = AttnGen_2[layer_idx_2]->get_duration();
        }
        else if(cur_2 == 2){
          pipe_2 = O_proj_2[layer_idx_2]->get_duration();
        }
        else {
          int expert_ffn_idx = layer_idx_2 / expert_freq;
          pipe_2 = ExpertFFN_2[expert_ffn_idx]->get_duration();
        }
      }
      else{ // is Non-MoE layer
        if(cur_2 == 0){
          pipe_2 = QKV_gen_2[layer_idx_2]->get_duration() + AttnSum_2[layer_idx_2]->get_duration();
        }
        else if(cur_2 == 1){
          pipe_2 = AttnGen_2[layer_idx_2]->get_duration();
        }
        else {
          int ffn_idx = layer_idx_2 - (layer_idx_2 / expert_freq) - 1;
          if(layer_idx_2 != num_layer - 1){
            pipe_2 = O_proj_2[layer_idx_2]->get_duration()
                    + FFN_2[ffn_idx]->get_duration()
                    + QKV_gen_2[layer_idx_2 + 1]->get_duration() + AttnSum_2[layer_idx_2 + 1]->get_duration();
          }
          else{
            pipe_2 = O_proj_2[layer_idx_2]->get_duration()
                    + FFN_2[ffn_idx]->get_duration();
          }
          layer_idx_2++;
          cur_2 = 0;
        }
      }
      stage_time += MAX(pipe_1, pipe_2);

      cur_1++;
      cur_2++;
      if(cur_1 == 4){
        layer_idx_1++;
        cur_1 = 0;
      }
      if(cur_2 == 4){
        layer_idx_2++;
        cur_2 = 0;
      }
    }
    time_list.push_back(stage_time);
    total_time += stage_time;
    std::cout << "Iter: " << std::to_string(i)
              << "\t| time: " << std::setprecision(4) << stage_time / 1000 << "us\t|";

    // check time
    scheduler1->updateScheduler();
    scheduler1->fillSequenceQueue();
    scheduler1->fillRunningQueue();

    scheduler2->updateScheduler();
    scheduler2->fillSequenceQueue();
    scheduler2->fillRunningQueue();

    // if(i != total_iter - 1){
    //   timeboard1.reset_timeboard();
    //   timeboard2.reset_timeboard();
    // }
  } 
  std::cout << "Total: " << std::to_string(total_time) << std::endl;
  // batch
  // auto metadata1 = scheduler1->setMetadata();
  // cluster1->run(metadata1);
  
  // scheduler1->updateScheduler();
  // scheduler1->fillSequenceQueue();
  // scheduler1->fillRunningQueue();
  
  cluster1->get_device(0)->top_module_graph->print_timeboard();
  cluster1->get_device(1)->top_module_graph->print_timeboard();
  // cluster1->get_device(2)->top_module_graph->print_timeboard();
  // cluster1->get_device(3)->top_module_graph->print_timeboard();

  cluster2->get_device(0)->top_module_graph->print_timeboard();
  cluster2->get_device(1)->top_module_graph->print_timeboard();
  // cluster2->get_device(2)->top_module_graph->print_timeboard();
  // cluster2->get_device(3)->top_module_graph->print_timeboard();

  // sort(time_list.begin(), time_list.end());
  // std::cout << "p50: " << time_list.at(p50) << ", p99: " << time_list.at(p99)
  //           << ", p99.9: " << time_list.at(p99dot9) << std::endl;
  return 0;
}

