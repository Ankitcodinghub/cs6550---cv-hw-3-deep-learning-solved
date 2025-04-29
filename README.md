# cs6550---cv-hw-3-deep-learning-solved
**TO GET THIS SOLUTION VISIT:** [CS6550 – CV HW 3: Deep Learning Solved](https://www.ankitcodinghub.com/product/cs6550-cv-hw-3-deep-learning-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;110512&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;2&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (2 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;CS6550 - CV HW 3: Deep Learning Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (2 votes)    </div>
    </div>
Problem A: Line Fitting

Download two csv files pA1.csv and pA2.csv using:

wget -nc 140.114.76.113:8000/pA1.csv wget -nc 140.114.76.113:8000/pA2.csv

You are asked to find the curve that fits the data using Stochastic Gradient Descent with a deep learning framework (PyTorch, Keras, etc). A1, A2 should be written in one Colab Notebook.

Left is pA1.csv and right is pA2.csv .

A1

Assume that the curve is y = f(x;a,b) = ax + b. You are asked to find the a,b that makes the curve best fit the data pA1.csv . Requirements:

1. A PyTorch code that outputs a and b. You get the points only if predicted a,b and ground truth a^,^b satisfies |a^ − a| ≤ 0.2 and |^b − b| ≤ 0.2. (8 points)

There’s a sample code here (https://colab.research.google.com/drive/1GZ_oX3PPwoDOamx0_XGbQHyt9ag1x5HM). Feel free to modify it. Hyperparameters(loss function, optimizer, batch size, epoch, …) are not restricted to the one used in sample code.

A2

You are asked to fit the data pA2.csv using following model:

y = f(x;w) = w0x2 + w1x + w2

where w are parameters.

Your code outputs w0,w1,w2. Similar to problem A1, you get the points only if predicted w and ground truth w^ satisfies |w0 − w^0| &lt; 0.2, |w1 − w^1| &lt; 0.2 and |w2 − w^2| &lt; 0.2. Requirements:

1. Use nn.Linear (PyTorch) / Dense (Keras or Tensorflow) to accomplish the task. (8 points)

2. A plot of loss against iteration or epoch. (7 points)

Problem B: License Plate Localization

Train Valid Test

Ground-truth are drawn in orange. Prediction are drawn in red.

Overview

For each image, there is one license plate. You are asked to localize the 4 corners of the license plate. That is, predict the (x,y) of each corner, 8 values in total. To reduce difficulties, you can fill in the blank of this reference code (https://colab.research.google.com/drive/1D4TYD73crnnyfvWa9-N-lpnsJlvRbhlC) to achieve the baseline.

Data

To download the data (317MB): wget -nc 140.114.76.113:8000/ccpd6000.zip

SHA256 checksum: 977d7124a53e565c3f2b371a871ee04ebbe572f07deb0b38c5548ddaae0cb2c9 Data is organized as:

ccpd6000/ train_images/ test_images/ train.csv sample.csv

There are 3000 images with annotation for training, 3000 images without label for testing. All images are taken from CCPD (https://github.com/detectRecog/CCPD).

Each row in train.csv has following fields:

1. name specifies the name of the image, full path is ccpd6000/train_images/&lt;name&gt;

2. BR_x , BR_y is the position of bottom-right corner

3. BL_x , BL_y is the position of bottom-left corner

4. TL_x , TL_y is the position of top-left corner

5. TR_x , TR_y is the position of top-right corner

The origin is at the top-left of the image.

sample.csv serves as a sample submission. Your submission should have the same format as sample.csv . Note that name is sorted in alphabetical order.

Evaluation

The metric is the root mean-square error between the predicted locations and the ground-truth locations of the 3000 testing images:

−−−−−−−−−−−−−−−−−−

RMSE = ⎷4N i=1 j=1 i i

where:

N is the number of images, j is the index of the corner, pji is the predicted location (x,y) of the j-th corner of image i. p^ji is the ground-truth location (x,y) of the j-th corner of image i.

To evaluate your prediction test.csv , use curl to POST the file to the server:

curl -F “file=@test.csv” -X POST 140.114.76.113:5000/cs6550 -i

Scoring

Baseline (40 points)

1. Training &amp; Validation (10 points)

2. Visualization of 25 training samples, 25 validation samples every epoch during training. (10 points)

3. Overlay training losses and validation losses in the same figure against step or epoch. (10 points)

4. Testing and RMSE ≤ 35.0 (10 points)

Your notebook should contain a cell that sends your prediction to the server, like the one shown in reference code (https://colab.research.google.com/drive/1D4TYD73crnnyfvWa9-N-lpnsJlvRbhlC#scrollTo=hE7xqUzkgmOl).

Improvement (20 points)

RMSE ≤ 20.0

Possible ways:

1. LR(learning rate) decay or lower LR.

2. Train longer (typically until the validation loss is converged).

3. Use deeper model, like ResNet18, to extract features.

4. Different optimizer, loss, etc.

5. Data augmentation.

6. Auxiliary task (learning), like segmentation.

7. Regress heatmaps instead of values, like in human pose estimation.

Bonus (10 points)

RMSE ≤ 15.0

You don’t need to copy the code many times to get improvement points and bonus points. You get the points automatically if your RMSE meet the criteria.

Report (10 points)

Describe the problems/difficulties you have faced in this homework and how you tackle them. The report report.pdf should be 2 pages at most.

Misc.

The structure of the turned-in file hw3_&lt;student_id&gt;.zip should be:

hw3/ pA.ipynb pB.ipynb report.pdf

&lt;student_id&gt; should be replaced with your student id, say hw3_17062566.zip . The zip file should be less than 5MB. pA.ipynb , pB.ipynb are notebooks that can run in Colab.

pA.ipynb contains the code of problem A1 and A2. pB.ipynb contains the code of problem B.

report.pdf describes the difficulties you have encountered.

The notebooks will be uploaded to Colab and executed by pressing “Runtime/Run All”.

Following conditions make you 0 point:

1. Code is not exectuable.

2. The archive format of the turned-in file is not zip.

3. The zip file does not follow the structure above.

4. Use external data other than the given data.

5. Code is not written by yourself.

6. Plagiarize.

7. Lookup the ground truth in the original dataset.

8. Label the images manually.

9. DoS attack the server.
