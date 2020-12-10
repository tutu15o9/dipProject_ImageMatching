# Image Classification and Matching Using Local Features and Homography


### Install the dependencies using the following command
```
pip install -m requirments.txt
```

Make the necessary changes in the main.py file change the path of template image and the query image in line 54 and line 55 respectively you can use sample images from the images forlder.

```
template_name = "./images/nemo_template.jpg"
query_name = "./images/nemo.jpg"
```

to 

```
template_name = "Template image path"
query_name = "Target image path"
```

Run the main file using command 
```
python main.py
```
## There will 2 new files
  #### 1 outline.png showing outlined image inside the target image.
  #### 2 linematches.png showing feature matching from template to target image(query image).
