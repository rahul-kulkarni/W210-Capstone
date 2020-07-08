docker run --rm --gpus '"device=2,3"' -it -p 8888:8888 -v $(pwd):/workspace -v /data:/data -v /storage:/storage bert_playground
