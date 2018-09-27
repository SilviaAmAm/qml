"""
This script shows how to set up the ARMP_G estimator where the xyz, nuclear charges, energies and forces are all set in
advance. Then the descriptors and the gradients are generated for all the samples. The indices are used to specify on
which samples to train/predict.

This script takes about 5 s to run.
"""

from qml.aglaia.aglaia import ARMP_G
import numpy as np
import os

## ------------- ** Loading the data ** ---------------

current_dir = os.path.dirname(os.path.realpath(__file__))
data = np.load(current_dir+"/../test/data/CN_isopentane_forces.npz")

xyz = data["arr_0"]
zs = data["arr_1"]
ene = data["arr_2"]
forces = data["arr_3"]

print(xyz.shape, zs.shape, ene.shape, forces.shape)

## ------------- ** Setting up the estimator ** ---------------

acsf_params={"nRs2":5, "nRs3":5, "nTs":5, "rcut":5, "acut":5, "zeta":220.127, "eta":30.8065}
estimator = ARMP_G(iterations=10, hidden_layer_sizes=(10, 5, 2), representation_name='acsf',
                   representation_params=acsf_params, tensorboard=False, store_frequency=1, batch_size=20)

estimator.set_xyz(xyz[:1])
estimator.set_classes(zs[:1])
estimator.set_properties(ene[:1])
estimator.set_gradients(forces[:1])

estimator.generate_representation(method="fortran")

## ----------- ** Fitting and predicting ** -------------------

idx = np.arange(0,1)

estimator.fit(idx)

score = estimator.score(idx)

print("The mean absolute error is %s (kcal/mol)." % (str(-score)))

energies_predict = estimator.predict(idx)