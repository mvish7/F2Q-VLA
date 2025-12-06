
# Model: 

* Which model to use?\
Qwen3-vl-2B came out on top in comparison. 

* Trainable token compression like Flex??\
Estimated 67B i.e. 150 - 300Millions data points are neeed to train a projector from scratch. SOOO its out of question

* DyCoke like token compression?\
Only works best for video-like scenario. are all/majority of our datasets videos-like??We can still use dycoke in multi-time stamp ,multi-image scenario but what to do in multi-image scenario?

* Any other token compression?\
Starts with finding suitable methods, implementing/adapting them for our use case. 


# SFT: 

* What's the purpose of my SFT?
  * Induce driving related knowledge and, spatial understanding (pointing, depth estimates)
  * introducing action head and enable action prediction

* What's the purose of RL?
  * Alignment with driving objective


# Datasets:
* Which datasets can i use for SFT?
* What's their nature wrt. images? i.e. single image, single-camera multi-timestamp, multi-camera single timestamp, multi-camera and multi-timestamp etc
* What type of annotations are provided?

# Evals:
* Which datasets are being used for evals in literature?
* Which dataset I can use for evals?
* What's the eval benchmark? i.e. metrics?? trajectory?? VQA?? etc??

# How to predict trajectory?
* SimLingo style? or Alpamayo style?? (HF VLA has diffusion implemented)




# How to RL?:
* Do I need a reward model? 
* Can I use Alpamayo as reward model? if yes how to use it for GRPO?
* How big of a GPU do I need for RL?

