# WindChaser
This is the DIRECT Project page for WindChaser group. This package is developed for wind power forecasting along with wind energy economic dispatch with a combination of LSTM and Q-Learning. The package was built and developed by wind data from wind farms in Seattle, WA. A GUI implementation for visualizing real-time forecasts is also available.

As the name suggests, we 'chase' the wind by focusing on the high penetration of stochastic power generation processes into our modern grids. We mainly have two objectives. One is to utilize deep learning for training a more accurate preditor that can forecasts future wind power generation based on history. Another one is to utilize reinforcement learning for helping users to make decisions based on the volatile electricity price to make informed decisions.

Group Member: Yiwen Wu, You Chen, Xiaoxiao Jia and Yize Chen

![alt text](https://github.com/yiwen26/WindChaser/blob/master/Graphs/Wind%20Power%20Forecasting%20(Without%20history%20power%20values%20input).png)

![alt text](https://github.com/yiwen26/WindChaser/blob/master/Graphs/1.png)

## License
This is currently a research project, and we do not plan to commercialize this, this project is under the permissive MIT license. If anything changes, we will be sure to update accordingly. If you do happen to want to use any parts of this project, please do give reference. For more details, please read LICENSE.md

## Language
Python 

Matlab

## Software Dependencies

* Python 3.5

* Tensorflow

* scikit-learn

* pandas

* numpy

* Matlab

## Repository Outlines

### Data
Contains the raw data related to wind power generation downloaded and sorted from <a href="https://www.nrel.gov/">`NREL`</a>. It also contains the processed data under the same directory that we used for machine learning algorithm in this project.

### Docs
Documentations about this project. This includes stand-up presentations, find poster and detailed API user documentation.

### GUI demo
The GUI created by `Matlab` shows the curve of the history ground truth of the wind power  as well as the prediction curve obtained from `LSTM` method of machine learning that we used in this project. Both the prediction and history power outages can be seen from the dashboard of this GUI.


### Graphs
The graph module includes all the workflow plot as well as data processing and analysis, LSTM forecasts, Q learning for decision making results.

### WindChaserModule
Main part of this project. Contains all python and Matlab packages, code and modules of machine learning method (both LSTM and Q learning) to implement the wind energy prediction.

### Poster
The browser will pop out pdf version of poster automatically.
<object data="http://blogs.uw.edu/yizechen/files/2018/03/Poster.pdf" type="application/pdf" width="700px" height="700px">
    <embed src="http://blogs.uw.edu/yizechen/files/2018/03/Poster.pdf">
        This browser does not support PDFs. Please download the PDF to view it: <a href="http://blogs.uw.edu/yizechen/files/2018/03/Poster.pdf">Download PDF</a>.</p>
    </embed>
</object>

You can also view our poster [here](https://github.com/yiwen26/WindChaser/blob/master/Docs/Presentation%20and%20posters/Poster.pdf)


### Q-Learning Part Presentation
You can also view our Q-Learning part introduction <a href="https://www.dropbox.com/s/avw5zm5t86dw7dq/Yize%20Part.mp4?dl=0">`video`</a>


