These codes emerged as a result of adapting the codes from Serengil's projects to our own project.

# deepface

<div align="center">

[![Downloads](https://static.pepy.tech/personalized-badge/deepface?period=total&units=international_system&left_color=grey&right_color=blue&left_text=downloads)](https://pepy.tech/project/deepface)
[![Stars](https://img.shields.io/github/stars/serengil/deepface?color=yellow&style=flat&label=%E2%AD%90%20stars)](https://github.com/serengil/deepface/stargazers)
[![License](http://img.shields.io/:license-MIT-green.svg?style=flat)](https://github.com/serengil/deepface/blob/master/LICENSE)
[![Tests](https://github.com/serengil/deepface/actions/workflows/tests.yml/badge.svg)](https://github.com/serengil/deepface/actions/workflows/tests.yml)
[![DOI](http://img.shields.io/:DOI-10.17671/gazibtd.1399077-blue.svg?style=flat)](https://doi.org/10.17671/gazibtd.1399077)

[![Blog](https://img.shields.io/:blog-sefiks.com-blue.svg?style=flat&logo=wordpress)](https://sefiks.com)
[![YouTube](https://img.shields.io/:youtube-@sefiks-red.svg?style=flat&logo=youtube)](https://www.youtube.com/@sefiks?sub_confirmation=1)
[![Twitter](https://img.shields.io/:follow-@serengil-blue.svg?style=flat&logo=x)](https://twitter.com/intent/user?screen_name=serengil)

[![Support me on Patreon](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Fshieldsio-patreon.vercel.app%2Fapi%3Fusername%3Dserengil%26type%3Dpatrons&style=flat)](https://www.patreon.com/serengil?repo=deepface)
[![GitHub Sponsors](https://img.shields.io/github/sponsors/serengil?logo=GitHub&color=lightgray)](https://github.com/sponsors/serengil)
[![Buy Me a Coffee](https://img.shields.io/badge/-buy_me_a%C2%A0coffee-gray?logo=buy-me-a-coffee)](https://buymeacoffee.com/serengil)

<!-- [![DOI](http://img.shields.io/:DOI-10.1109/ICEET53442.2021.9659697-blue.svg?style=flat)](https://doi.org/10.1109/ICEET53442.2021.9659697) -->
<!-- [![DOI](http://img.shields.io/:DOI-10.1109/ASYU50717.2020.9259802-blue.svg?style=flat)](https://doi.org/10.1109/ASYU50717.2020.9259802) -->

</div>

<p align="center"><img src="https://raw.githubusercontent.com/serengil/deepface/master/icon/deepface-icon-labeled.png" width="200" height="240"></p>

DeepFace is a lightweight [face recognition](https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/) and facial attribute analysis ([age](https://sefiks.com/2019/02/13/apparent-age-and-gender-prediction-in-keras/), [gender](https://sefiks.com/2019/02/13/apparent-age-and-gender-prediction-in-keras/), [emotion](https://sefiks.com/2018/01/01/facial-expression-recognition-with-keras/) and [race](https://sefiks.com/2019/11/11/race-and-ethnicity-prediction-in-keras/)) framework for python. It is a hybrid face recognition framework wrapping **state-of-the-art** models: [`VGG-Face`](https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/), [`FaceNet`](https://sefiks.com/2018/09/03/face-recognition-with-facenet-in-keras/), [`OpenFace`](https://sefiks.com/2019/07/21/face-recognition-with-openface-in-keras/), [`DeepFace`](https://sefiks.com/2020/02/17/face-recognition-with-facebook-deepface-in-keras/), [`DeepID`](https://sefiks.com/2020/06/16/face-recognition-with-deepid-in-keras/), [`ArcFace`](https://sefiks.com/2020/12/14/deep-face-recognition-with-arcface-in-keras-and-python/), [`Dlib`](https://sefiks.com/2020/07/11/face-recognition-with-dlib-in-python/), `SFace` and `GhostFaceNet`.

[`Experiments`](https://github.com/serengil/deepface/tree/master/benchmarks) show that **human beings have 97.53% accuracy** on facial recognition tasks whereas those models already reached and passed that accuracy level.

## Installation [![PyPI](https://img.shields.io/pypi/v/deepface.svg)](https://pypi.org/project/deepface/)

The easiest way to install deepface is to download it from [`PyPI`](https://pypi.org/project/deepface/). It's going to install the library itself and its prerequisites as well.

```shell
$ pip install deepface
```

Alternatively, you can also install deepface from its source code. Source code may have new features not published in pip release yet.

```shell
$ git clone https://github.com/serengil/deepface.git
$ cd deepface
$ pip install -e .
```

Once you installed the library, then you will be able to import it and use its functionalities.

```python
from deepface import DeepFace
```
