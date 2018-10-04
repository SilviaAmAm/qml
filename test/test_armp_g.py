# MIT License
#
# Copyright (c) 2018 Silvia Amabilino
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
This test checks if all the ways of setting up the estimator ARMP work.
"""

import numpy as np
from qml.aglaia.aglaia import ARMP_G
from qml.utils.utils import InputError
import glob
from qml.utils.utils import is_array_like
import os
import shutil
import tensorflow as tf

def test_set_representation():
    """
    This function tests the function _set_representation.
    """
    try:
        ARMP_G(representation_name='acsf', representation_params={'slatm_sigma12': 0.05})
        raise Exception
    except InputError:
        pass

    try:
        ARMP_G(representation_name='coulomb_matrix')
        raise Exception
    except InputError:
        pass

    try:
        ARMP_G(representation_name='slatm')
        raise Exception
    except InputError:
        pass

    parameters = {'rcut': 10.0, 'acut': 10.0, 'nRs2': 3, 'nRs3': 3, 'nTs': 2,
                                      'zeta': 3.0, 'eta': 2.0}

    estimator = ARMP_G(representation_name='acsf', representation_params=parameters)

    assert estimator.representation_name == 'acsf'

    for key, value in estimator.acsf_parameters.items():
        if is_array_like(value):
            assert np.all(estimator.acsf_parameters[key] == parameters[key])
        else:
            assert estimator.acsf_parameters[key] == parameters[key]

def test_set_properties():
    """
    This test checks that the set_properties function sets the correct properties.
    :return:
    """
    test_dir = os.path.dirname(os.path.realpath(__file__))

    energies = np.loadtxt(test_dir + '/CN_isopentane/prop_kjmol_training.txt',
                          usecols=[1])

    estimator = ARMP_G()

    assert estimator.properties == None

    estimator.set_properties(energies)

    assert np.all(estimator.properties == energies)

def test_set_representation_and_dgdr():
    """
    This test checks that the set_representation function works as expected.
    :return:
    """
    test_dir = os.path.dirname(os.path.realpath(__file__))

    data_incorrect = np.load(test_dir + "/data/CN_isopent_light_UCM.npz")
    data_correct = np.load(test_dir + "/data/local_slatm_ch4cn_light.npz")
    representation_correct = np.asarray(data_correct["arr_0"])
    representation_incorrect = np.asarray(data_incorrect["arr_0"])

    dgdr_correct = np.ones((3, 4, 5, 4, 3))
    dgdr_incorrect = np.ones((3, 1, 2, 3, 4))


    estimator = ARMP_G()

    assert estimator.g == None

    estimator._set_representation(g=representation_correct)

    assert np.all(estimator.g == representation_correct)

    assert estimator.dg_dr == None

    estimator.set_dgdr(dgdr_correct)

    assert np.all(estimator.dg_dr == dgdr_correct)

    # Pass a representation with the wrong shape
    try:
        estimator._set_representation(g=representation_incorrect)
        raise Exception
    except InputError:
        pass

    # Pass a dgdr with the wrong shape
    try:
        estimator.set_dgdr(dgdr_incorrect)
        raise Exception
    except InputError:
        pass

def test_fit_1():
    """
    This function tests the first way of fitting the representation: the data is passed by first creating compounds and then
    the representations are created from the compounds.
    """
    tf.reset_default_graph()
    test_dir = os.path.dirname(os.path.realpath(__file__))

    filenames = glob.glob(test_dir + "/CN_isopentane/*.xyz")
    energies = np.loadtxt(test_dir + '/CN_isopentane/prop_kjmol_training.txt',
                          usecols=[1])
    data = np.load(test_dir + "/data/CN_isopentane_forces.npz")
    filenames.sort()
    forces =  data["arr_3"][:2]

    estimator = ARMP_G(representation_name="acsf")
    estimator.generate_compounds(filenames[:2])
    estimator.set_properties(energies[:2].astype(np.float32))
    estimator.set_gradients(forces.astype(np.float32))

    idx = np.arange(0, 2)
    estimator.fit(idx)
    estimator.score(idx)

def test_predict_fromxyz():
    """
    This test checks that the predictions from the "predict" and the "predict_from_xyz" functions are the same.
    It also checks that if the model is saved, when the model is reloaded the predictions are still the same.
    """

    tf.reset_default_graph()

    xyz = np.array([[[0, 1, 0], [0, 1, 1], [1, 0, 1]],
           [[1, 2, 2], [3, 1, 2], [1, 3, 4]],
           [[4, 1, 2], [0.5, 5, 6], [-1, 2, 3]]], dtype=np.float32)
    zs = np.array([[1, 2, 3],
          [1, 2, 3],
          [1, 2, 3]], dtype=np.int32)

    ene_true = np.array([0.5, 0.9, 1.0], dtype=np.float32)
    forces_true = np.array([[[0, 1, 0], [0, 1, 1], [1, 0, 1]],
           [[1, 2, 2], [3, 1, 2], [1, 3, 4]],
           [[4, 1, 2], [0.5, 5, 6], [-1, 2, 3]]], dtype=np.float32)

    acsf_param = {"nRs2": 5, "nRs3": 5, "nTs": 5, "rcut": 5, "acut": 5, "zeta": 220.127, "eta": 30.8065}
    estimator = ARMP_G(iterations=10, l1_reg=0.0001, l2_reg=0.005, learning_rate=0.0005, representation_name='acsf',
                     representation_params=acsf_param)

    estimator.set_xyz(xyz)
    estimator.set_classes(zs)
    estimator.set_properties(ene_true)
    estimator.set_gradients(forces_true)

    idx = list(range(xyz.shape[0]))

    estimator.fit(idx)

    ene1, f1 = estimator.predict(idx)
    ene2, f2 = estimator.predict_from_xyz(xyz, zs)

    assert np.all(np.isclose(ene1, ene2, rtol=1.e-6))

    estimator.save_nn(save_dir="temp")

    new_estimator = ARMP_G(iterations=10, l1_reg=0.0001, l2_reg=0.005, learning_rate=0.0005, representation_name='acsf',
                         representation_params=acsf_param)

    new_estimator.load_nn(save_dir="temp")

    new_estimator.set_xyz(xyz)
    new_estimator.set_classes(zs)
    new_estimator.set_properties(ene_true)
    new_estimator.set_gradients(forces_true)

    ene3, f3 = new_estimator.predict(idx)
    ene4, f4 = new_estimator.predict_from_xyz(xyz, zs)

    shutil.rmtree("temp")
    os.remove("predict.tfrecords")
    os.remove("training.tfrecords")

    assert np.all(np.isclose(ene3, ene4, rtol=1.e-6))
    assert np.all(np.isclose(ene1, ene3, rtol=1.e-6))

if __name__ == "__main__":
    test_set_representation()
    test_set_properties()
    test_set_representation_and_dgdr()
    test_fit_1()
    test_predict_fromxyz()
