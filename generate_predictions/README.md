# Note the notebook refers to local files not in the repository

## Docker Stuff
### Build docker  
From within the `generate_predictions` directory run:  
```
$ docker build -t w210_jupyter .
```
Or run the `docker_build.sh` on Linux/Mac. 


### Run Docker
From within the `generate_predictions` directory run:   

**Note:** Below instructions assume no GPU, to run Docker with GPU, include `--gpus=all` flag. 

#### On Linux/Mac
```
$ docker run --rm -it -p 8888:8888 -v $(pwd):/workspace w210_jupyter 
```
Or run `run_docker.sh` script.  


#### On Windows
```
> docker run --rm -it -p 8888:8888 -v ${PWD}:/workspace w210_jupyter 
```

Once the docker container is running, start Jupyter:  
Winthin the `/workspace` directory (default) run:   
```
$ nohup jupyter notebook --ip=0.0.0.0 --allow-root &  
$ tail nohup.out  
```
There is also an alias `jupnote` for the above command.   
