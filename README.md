# Image Analysis II - Project

This repository contains the implementation of a Deep Convolutional Neural Network (CNN) for classifying cervical cancer on cell images.

The project was carried as part of Eléctronique et Informatique Industrielle (EII) course of Institut National des Sciences Appliquées (INSA) de Rennes, Image Analysis II.

__Authors__ Eduardo Fernandes Montesuma, Tudy Gourmelen, Timothée Saint-Cast

Our implementation was based on the [1]. The CNN was trained using the Herlev [Pap-Smear database](http://mde-lab.aegean.gr/index.php/downloads), which is pubicly available by the Herlev University Hospital [2].

# Executing the project

This project was designed as a Python package. All its dependencies are listed in _requirements.txt_. By default Tensorflow gpu is enabled, however, if you want to run it by cpu, you should modify the _requirements.txt_ entry from "_tensorflow-gpu==1.15.0_" to "_tensorflow==1.15.0_". This, however, is not advised since training and prediction can take long time on CPU. We suppose you have anaconda installed on your machine, or at lest virtual environments.

To execute the project, you first need to create a virtual environment:

```bash
$ conda create --name ProjAnim python=3.6
```

Once your virtual environment is created, you may activate it by using:

```bash
$ conda activate ProjAnim
```

Then, the dependencies are installed using:

```bash
$ pip install -r requirements.txt
```

Once all dependencies are installed, you may run a training session by using:

```bash
$ python run_train.py
```

Or a evaluation session by usnig:

```bash
$ python run_test.py
```

We, however, advise you to perform data augmentation on your images beforehand, placing them on "./data/database/tmp". Beware that the complete dataset, before augmentation, may took up to 6GB of memory. The process may take a few minutes.

To augment the images on the dataset, simply go to "./scripts" and run "augment_images.py".


# References

[1] Zhang, L., Lu, L., Nogues, I., Summers, R.M., Liu, S. and Yao, J., 2017. DeepPap: deep convolutional networks for cervical cell classification. IEEE journal of biomedical and health informatics, 21(6), pp.1633-1643.

[2] Jantzen, J., Norup, J., Dounias, G. and Bjerregaard, B., 2005. Pap-smear benchmark data for pattern classification. Nature inspired Smart Information Systems (NiSIS 2005), pp.1-9.



