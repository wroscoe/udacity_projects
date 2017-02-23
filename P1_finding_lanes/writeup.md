### Reflection
#### The Pipeline
There were two major parts of this project. 
1. Find the lane markings.
2. Generalize the lane markings into a solid line and show them on the video. 

To find the lane markings, I used the following steps:
1. Increase the contrast of the white and yellow colors to highlight the lines.
2. Blurr the image to hide small edges.
3. Apply canny edge detection to find the lines.
3. Eliminate edges outside a region of interest.
5. Find lines using hough transform.

One the lines of the image have been found we have to generalize them to create a continous lines to deliminate the lane. To do this I used the following steps.
1. Converte lines from two points to slope and y intercept.
2. Filter out lines outside the range of resonable slopes for lanes (<.5 and >2)
3. Separate lane markings into a left and right side.
4. Average slopes of lines.
5. Draw lines from top of lane to bottom of picture.

### Possible shortcomings
1. In a few frames of the video one side of the lanes is not seen so no line is shown. 
2. The lines tend to jump around and don't have smooth transitions from one frame to the next.
3. The straign lines can't represent curved lanes.
4. Though not seen in this highway footage, sharp turns would break my approach because both lines could have positive or negative slopes. In this case the average of both lines would be drawn. 

### Improvements
This approach could be improved by time weighting the lines so that past line angles would smooth the possitions of lines between frames. This would also show a line even when a frame didn't pick one up. 

Clustering the hough lines rather than dividing them between possitive slopes would allow for the discovery of the lines during sharp turns. 
