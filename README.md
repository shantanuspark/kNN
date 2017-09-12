# kNN
kNN implementation in Python

Following steps should be performed before running the code:
1. The correct training data set (any 10-dobscv) should be downloaded from <a href="http://sci2s.ugr.es/keel/category.php?cat=clas">keel</a>. 
Any dataset would work (recommended is heart or banana, have included them in the repo).
Extract the zip and copy the data folder besides the shantanu_deshmukh_knn.py file 

2. For plotting the graph, I have used matplotlib, so to install it use ->
sudo apt-get install python-matplotlib

That’s it, now simply RUN the program..


Highlights of the program :
1. Automates the tasks to study how accuracy depends on the value of k.
2. Can run any 10-dobscv dataset on Keel
3. Plot graphs do illustrate accuracy for different datasets and K values
4. Intelligently use information contained in the @attribute field to select the most appropriate
distance metric
5. There is an option to run the program directly using command line arguments besides the command line menu that appears on running the program directly

Output of the program -
Accuracy vs value of K graph:
<img src="https://drive.google.com/open?id=0B9GyGROQo3hiQWtoU0FfNlM0aXM" />

10-fold Cross Validation 
Accuracy vs iteration graph
<img src="https://drive.google.com/open?id=0B9GyGROQo3hiMHluTEtkanhpaWM" >

Refer the README pdf for more detailed steps with screenshots on running the program

Datasets used from - http://sci2s.ugr.es/keel/category.php?cat=clas
Thankyou Keel for the datasets..
