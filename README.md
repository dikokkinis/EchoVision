# EchoVision 🔊👁️
> Zero-shot sound source localization in video using
> cross-modal attention between CLAP and CLIP.
> No training. No labels. Just geometry.

Of course the result is nowhere near trained models, since the embedding spaces of CLIP and CLAP are not aligned. This is just a side project, zero-shot, with no training. 

## What it does
The input to the model is a video clip with distinct sound sources. The output is the video with a heatmap overlay on, that shows the intensity of the sound relevance inside the frame of the video.

## How it works
Basically, the framework computes a cosine similarity for every frame between the CLIP image embedding and the CLAP audio embedding, where both models where trained with contrastive loss between text and their respective modality. 

**Try the notebook tutorial!**: It shows step by step the process `notebooks/echovision_tutorial.ipynb`

## Local Use
Download and install `ffmpeg`
### Gradio Interface
First clone the repo and then run the following commands

    pip install -r requirements.txt  
    cd EchoVision
    python app.py

### Run inference on one video sample
The video can be either a path to an already downloaded video or you can provide a valid url.

    python run_simple.py --video_path your/video/path --output_dir your/output/dir

