# ISIC 2019 - Skin Lesion Analysis Towards Melanoma Detection

This is a [Keras](https://keras.io) with [TensorFlow](https://www.tensorflow.org/) backend implementation for [ISIC 2019 Challenge](https://challenge2019.isic-archive.com) Task 1: classify dermoscopic images among nine different diagnostic categories without meta-data.

## Getting Started

### Dependencies

* Python 3.5 or above
* Keras 2.2.4
* TensorFlow 1.14
* [pandas](https://pandas.pydata.org)
* [NumPy](https://www.numpy.org)
* [Matplotlib](https://matplotlib.org)
* [scikit-learn](https://scikit-learn.org)
* [OpenCV-Python](https://github.com/skvark/opencv-python)
* [tqdm](https://github.com/tqdm/tqdm)
* [Augmentor](https://github.com/mdbloice/Augmentor): A modification of Augmentor 0.2.3 is under [Augmentor](Augmentor/)

### Datasets

* [ISIC 2019 Training Data](https://challenge2019.isic-archive.com/data.html#training-data)
* [ISIC Archive](https://www.isic-archive.com/#!/topWithHeader/onlyHeaderTop/gallery)

| Diagnostic Category                | Amount |
| ---------------------------------- | ------ |
| Angiofibroma or fibrous papule     | 1      |
| Angioma                            | 12     |
| Atypical melanocytic proliferation | 12     |
| Lentigo NOS                        | 70     |
| Lentigo simplex                    | 22     |
| Scar                               | 1      |

* [Seven-Point Checklist Dermatology Dataset](http://derm.cs.sfu.ca/)

After downloading the dataset, all melanosis images can be retrieved from the dataset by using the [notebook](derm7pt.ipynb).

| Diagnostic Category | Amount |
| ------------------- | ------ |
| Melanosis           | 16     |
