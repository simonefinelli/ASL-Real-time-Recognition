# American Sing Language Real-time Recognition
The following project aims to demonstrate the feasibility of translating
American Sign Language with a real-time approach.

The results obtained from the study of the problem are contained in the
real-time demo application. Below, some GIFs are extracted from the webcam
video stream.

<div align="center">
  <img src="./media/book.gif" width="275px" />
  <img src="./media/computer.gif" width="275px" /> 
  <img src="./media/drink.gif" width="275px" />
</div>

<br> In addition, a client-server web app has also been implemented.
The following GIF shows how it works. <br>

<div align="center">
<img src="./media/webapp.gif" width="80%" />
</div>


## Getting Started
Follow the instructions below to get a clean installation.

### Dataset
Download the WLASL dataset.
```sh
git clone https://github.com/dxli94/WLASL
```

### Prerequisites
Create and activate a new virtual environment in the project folder.
```sh
~/project_folder$ virtualenv .env
~/project_folder$ source .env/bin/activate
```

### Installation
1. Clone the repo.
   ```sh
   (.env) git clone https://github.com/simonefinelli/ASL-Recognition-backup
   ```
2. Install requirements.
   ```sh
   (.env) python -m pip install -r requirements.txt
   ```
3. Split the WLASL dataset in the right format using the script in
   'tools/dataset splitting/'.
   ```sh
   (.env) python k_gloss_splitting.py ./WLASL_full/ 2000
   ```
4. Copy the pre-processed dataset in the 'data' folder.


## Usage
Now let's see how to use the neural network, the demo and the web app.

### Neural Net
1. To start the training run:
   ```sh
   (.env) python train_model.py
   ```
2. After training, to evaluate the best model on the test-set, run:
   ```sh
   (.env) python evaluate_model.py
   ```
3. Now, we can use the model in the demo or for the web app.

#### Tips
* The WLASL dataset can be divided into 4 sub-datasets: WLASL100, WLASL300,
WLASL1000 e WLASL2000. You can find the various models used for each sub-dataset
in the models.py file.
* The custom frame generator used in the model needs at least 12 frames
to work. However, videos 59958, 18223, 15144, 02914 and 55325, in the
WLASL1000 and WLASL2000 datasets, are shorter. To solve this problem use the
video_extender.py script.


### Real-time demo
1. To start the demo run:
   ```sh
   (.env) python demo.py
   ```

### Web app
1. To start the web app run:
   ```sh
   (.env) python serve.py
   ```
2. Go to the following URL: [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

#### Tip
The model used in the demo and the web app was obtained by training
the neural net on a custom dataset, called WLASL20custom. This dataset
consists of only 20 words: book, chair, clothes, computer, drink, drum,
family, football, go, hat, hello, kiss, like, play, school, street, table,
university, violin and wall.


## Results
I achieved the following accuracy with the proposed models:
1. WLASL20c:  63% of accuracy.
2. WLASL100:  34% of accuracy.
3. WLASL300:  28% of accuracy.
4. WLASL1000: 19% of accuracy.
5. WLASL2000: 10% of accuracy.


## License
Distributed under the MIT License. See `LICENSE` for more information.


## Acknowledgements
* [WLASL dataset](https://dxli94.github.io/WLASL/)
* [Data generators with Keras](https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly)
* [Video generator approach](https://github.com/peachman05/action-recognition-tutorial/blob/master/data_gen.py)
