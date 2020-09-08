# ImagePanorama

The aim of this project was to implement Image stitching methods by calculating homographies of the two images. 
And then use the created homography matrix to stitch two images resulting in a panorama.

The first step is to find out the common area between the images and then create a map of the best matches between the images. The matches are the pixels that have the most commonalities. Then using these pixels a matrix is calculated that maps pixel of one image onto the other. Then this matrix is applied on the whole image that results in a panorama.
<br/> <br/>
We can see the matches as below:
<br/> <br/>
<img src="https://github.com/saini-vishal/ImagePanorama/blob/master/test/Matches.jpg" width=700/>

These are the total number of matches found, out of these matches, using a threshold, the best matches are filtered.
<br/> <br/>
<img src="https://github.com/saini-vishal/ImagePanorama/blob/master/test/Good_Matches.jpg" width=700/>

These filtered matches are used for the further process.


A few results of the implemented code are as below:

<img align="left" src="https://github.com/saini-vishal/ImagePanorama/blob/master/data/mountain/mountain1.jpg" width=250/> <img align="center" src="https://github.com/saini-vishal/ImagePanorama/blob/master/data/mountain/mountain2.jpg" width=250/>

As you can see there are two differnet images with some area of the image common. Now after running the script, the result obtained is:
<br/>
<img src="https://github.com/saini-vishal/ImagePanorama/blob/master/data/mountain/panorama.jpg" width=500/>
<br/> <br/> <br/>
Similarly a few of the results are:
<br/> <br/> <br/>
<img src="https://github.com/saini-vishal/ImagePanorama/blob/master/data/ubdata/ub1.jpg" width=200/> <img src="https://github.com/saini-vishal/ImagePanorama/blob/master/data/ubdata/ub2.jpg" width=200 /> <img src="https://github.com/saini-vishal/ImagePanorama/blob/master/data/ubdata/ub3.jpg" width=200 /> <img src="https://github.com/saini-vishal/ImagePanorama/blob/master/data/ubdata/ub4.jpg" width=200/>
<br/> <br/> <br/>
The resulting panorama:
<br/> <br/> <br/>
<img src="https://github.com/saini-vishal/ImagePanorama/blob/master/data/ubdata/panorama.jpg" width=500 />
