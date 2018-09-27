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
This file contains tests for the tensorflow atom centred symmetry function module. It uses the numpy implementation
as a comparison.
"""

import tensorflow as tf
import numpy as np

import qml.aglaia.symm_funct as symm_funct
import qml.aglaia.np_symm_funct as np_symm_funct

def test_acsf_1():
    """
    This test compares the atom centred symmetry functions generated with tensorflow and numpy.
    The test system consists of 5 configurations of CH4 + CN radical.
    :return: None
    """

    nRs2 = 3
    nRs3 = 3
    nTs = 3
    rcut = 5
    acut = 5
    zeta = 220.127
    eta = 30.8065

    input_data = "/Volumes/Transcend/repositories/my_qml_fork/qml/test/data/data_test_acsf.npz"
    data = np.load(input_data)

    xyzs = data["arr_0"]
    zs = data["arr_1"]
    elements = data["arr_2"]
    element_pairs = data["arr_3"]

    n_samples = xyzs.shape[0]
    n_atoms = zs.shape[1]

    with tf.name_scope("Inputs"):
        zs_tf = tf.placeholder(shape=[n_samples, n_atoms], dtype=tf.int32, name="zs")
        xyz_tf = tf.placeholder(shape=[n_samples, n_atoms, 3], dtype=tf.float32, name="xyz")

    acsf_tf_t = symm_funct.generate_parkhill_acsf(xyz_tf, zs_tf, elements, element_pairs, rcut, acut, nRs2, nRs3, nTs, zeta, eta)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    acsf_tf = sess.run(acsf_tf_t, feed_dict={xyz_tf: xyzs, zs_tf: zs})

    acsf_np = np_symm_funct.generate_acsf(xyzs, zs, elements, element_pairs, rcut, acut, nRs2, nRs3, nTs, zeta, eta)

    n_samples = xyzs.shape[0]
    n_atoms = xyzs.shape[1]

    for i in range(n_samples):
        for j in range(n_atoms):
            acsf_np_sort = np.sort(acsf_np[i][j])
            acsf_tf_sort = np.sort(acsf_tf[i][j])
            np.testing.assert_array_almost_equal(acsf_np_sort, acsf_tf_sort, decimal=4)

def test_acsf_2():
    """
    This test compares the atom centred symmetry functions generated with tensorflow and numpy.
    The test system consists of 10 molecules from the QM7 data set.
    :return: None
    """
    nRs2 = 3
    nRs3 = 3
    nTs = 3
    rcut = 5
    acut = 5
    zeta = 220.127
    eta = 30.8065

    input_data = "/Volumes/Transcend/repositories/my_qml_fork/qml/test/data/qm7_testdata.npz"
    data = np.load(input_data)

    xyzs = data["arr_0"]
    zs = data["arr_1"]
    elements = data["arr_2"]
    element_pairs = data["arr_3"]

    n_samples = xyzs.shape[0]
    max_n_atoms = zs.shape[1]

    with tf.name_scope("Inputs"):
        zs_tf = tf.placeholder(shape=[n_samples, max_n_atoms], dtype=tf.int32, name="zs")
        xyz_tf = tf.placeholder(shape=[n_samples, max_n_atoms, 3], dtype=tf.float32, name="xyz")

    acsf_tf_t = symm_funct.generate_parkhill_acsf(xyz_tf, zs_tf, elements, element_pairs, rcut, acut, nRs2, nRs3, nTs, zeta, eta)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    acsf_tf = sess.run(acsf_tf_t, feed_dict={xyz_tf: xyzs, zs_tf: zs})

    acsf_np = np_symm_funct.generate_acsf(xyzs, zs, elements, element_pairs, rcut, acut, nRs2, nRs3, nTs, zeta, eta)

    for i in range(n_samples):
        for j in range(max_n_atoms):
            if zs[i][j] == 0:
                continue
            else:
                acsf_np_sort = np.sort(acsf_np[i][j])
                acsf_tf_sort = np.sort(acsf_tf[i][j])
                np.testing.assert_array_almost_equal(acsf_np_sort, acsf_tf_sort, decimal=4)

def test_acsf_3():
    """
    This test compares the one at a time tensorflow predictions to the numpy predictions.
    """

    nRs2 = 3
    nRs3 = 3
    nTs = 3
    rcut = 5
    acut = 5
    zeta = 220.127
    eta = 30.8065

    input_data = "/Volumes/Transcend/repositories/my_qml_fork/qml/test/data/data_test_acsf.npz"
    data = np.load(input_data)

    xyzs = data["arr_0"]
    zs = data["arr_1"]
    elements = data["arr_2"]
    element_pairs = data["arr_3"]

    n_samples = xyzs.shape[0]
    n_atoms = zs.shape[1]

    with tf.name_scope("Inputs_G"):
        zs_tf = tf.placeholder(shape=[n_samples, n_atoms], dtype=tf.int32, name="Classes")
        xyz_tf = tf.placeholder(shape=[n_samples, n_atoms, 3], dtype=tf.float32, name="xyz")

        dataset = tf.data.Dataset.from_tensor_slices((xyz_tf, zs_tf))
        iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
        batch_xyz, batch_zs = iterator.get_next()

    with tf.name_scope("Descriptor"):
        representation = symm_funct.generate_parkhill_acsf_single(xyzs=batch_xyz, Zs=batch_zs, elements=elements,
                                                       element_pairs=element_pairs, rcut=rcut, acut=acut,
                                                       nRs2=nRs2, nRs3=nRs3, nTs=nTs, eta=eta, zeta=zeta)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(iterator.make_initializer(dataset), feed_dict={xyz_tf: xyzs, zs_tf: zs})

    tf_acsf = []

    while True:
        try:
            tf_acsf.append(sess.run(representation))
        except tf.errors.OutOfRangeError:
            break

    tf_acsf = np.asarray(tf_acsf)

    acsf_np = np_symm_funct.generate_acsf(xyzs, zs, elements, element_pairs, rcut, acut, nRs2, nRs3, nTs, zeta, eta)

    for i in range(n_samples):
        for j in range(n_atoms):
            acsf_np_sort = np.sort(acsf_np[i][j])
            acsf_tf_sort = np.sort(tf_acsf[i][j])
            np.testing.assert_array_almost_equal(acsf_np_sort, acsf_tf_sort, decimal=4)

if __name__ == "__main__":
    test_acsf_1()
    test_acsf_2()
    test_acsf_3()