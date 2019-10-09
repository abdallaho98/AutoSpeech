AutoSpeech
======================================

THE ORIGINAL VERSION IS FROM https://github.com/mortal123/autonlp_starting_kit,
MODIFIED BY SHOUXIANG LIU.

## Contents
ingestion/: The code and libraries used on Codalab to run your submmission.

scoring/: The code and libraries used on Codalab to score your submmission.

code_submission/: An example of code submission you can use as template.

sample_data/: Some sample data to test your code before you submit it.

run_local_test.py: A python script to simulate the runtime in codalab

## Local development and testing
1. Download the `autospeech-sample-data.zip` and you will get a directory called `"DEMO"` after unzipping it.
2. Put the `DEMO` directory into the `sample_data` directory.
You can download the sample data from the [challenge website](https://autodl.lri.fr/competitions/48#learn_the_details).
3. To make your own submission to AutoSpeech challenge, you need to modify the
file `model.py` in `code_submission/`, which implements your algorithm.
4. Test the algorithm on your local computer using Docker,
in the exact same environment as on the CodaLab challenge platform. Advanced
users can also run local test without Docker, if they install all the required
packages.
5. If you are new to docker, install docker from https://docs.docker.com/get-started/.
Then, at the shell, run:
```
cd path/to/autospeech_starting_kit/
(CPU) docker run -it -v "$(pwd):/app/codalab" nehzux/autospeech:gpu
(GPU) docker run --gpus '"device=0"' -it -v "$(pwd):/app/codalab" nehzux/autospeech:gpu</strong></p>
```
Please note that for running docker with GPU support, you need to instal [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) first.
The option `-v "$(pwd):/app/codalab"` mounts current directory
(`autospeech_starting_kit/`) as `/app/codalab`. If you want to mount other
directories on your disk, please replace `$(pwd)` by your own directory.

The Docker image
```
nehzux/autospeech:gpu
```
has Nvidia GPU supports. see the
[site](https://hub.docker.com/r/nehzux/autospeech)
to check installed packages in the docker image.

Make sure you use enough RAM (**at least 4GB**).

6. You will then be able to run the `ingestion program` (to produce predictions)
and the `scoring program` (to evaluate your predictions) on toy sample data.
In the AutoSpeech challenge, both two programs will run in parallel to give
real-time feedback (with learning curves). So we provide a Python script to
simulate this behavior. To test locally, run:
```
python run_local_test.py
```
Then you can view the real-time feedback with a learning curve by opening the
HTML page in `scoring_output/`.

The full usage is
```
python run_local_test.py -dataset_dir=./sample_data/DEMO -code_dir=./code_submission
```
You can change the argument `dataset_dir` to other datasets (e.g. the five
practice datasets we provide). On the other hand,
you can also modify the directory containing your other sample code
(`model.py`).

## Download practice datasets
We provide 5 practice datasets for participants. They can use these datasets to:
1. Do local test for their own algorithm;
2. Enable meta-learning.

You may refer to [codalab site](https://autodl.lri.fr/competitions/48#learn_the_details-get_data) for practice datasets.

Unzip the zip file and you'll get 5 datasets.

## Prepare a ZIP file for submission on CodaLab
Zip the contents of `code_submission`(or any folder containing
your `model.py` file) without the directory structure:
```
cd code_submission/
zip -r mysubmission.zip *
```
then use the "Upload a Submission" button to make a submission to the
competition page on CodaLab platform.

Tip: to look at what's in your submission zip file without unzipping it, you
can do
```
unzip -l mysubmission.zip
```

## Report bugs and create issues

If you run into bugs or issues when using this starting kit, please create
issues on the
[*Issues* page](https://github.com/liushouxiang/autospeech_starting_kit/issues)
of this repo. Two templates will be given when you click the **New issue**
button.

## Contact us
If you have any questions, please contact us via:
<autospeech2019@4paradigm.com>
