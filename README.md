# Where am I?: Scene Retrieval with Language

This is the repository that contains source code for the work [Where am I?: Scene Retrieval with Language](https://whereami-langloc.github.io/).

# Evaluation
First download the model weights from [here](https://drive.google.com/file/d/1Ol1LuPVIVXvSXPmuMoEc5sIg20fCJ_su/view?usp=sharing) and place it in `/playground/graph_models/model_checkpoints/graph2graph/`.
Then run the `run_eval.sh` script in `/shell/`.

# Training
Run the `run.sh` script in `/shell/`.

# Baselines
The CLIP2CLIP baseline can be found and run in the `/baselines/CLIP2CLIP/` folder. And the Text2Pos baseline can be found in this [fork](https://github.com/jiaqchen/Text2Pos-CVPR2022). 
The model weights for the fine-tuned version of Text2Pos can be found [here](https://drive.google.com/file/d/1Bkev47FdHgiLFF2-4BOMhp4P8W0ZOnNh/view?usp=sharing), and for the version trained from scratch on the 3DSSG dataset is [here](https://drive.google.com/file/d/1gJUF9Tgdket1ebu8gQsJyrd59MN3VJI8/view?usp=sharing).
To run the Text2Pos models, you can run `run_text2pos.sh` for training and `run_eval_text2pos.sh` for evaluation.
