import scipy
from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt


def load_and_center_dataset(filename):
    # Load file into python, which gives a nxd dataset (421x4096)
    x = np.load(filename)
    # Find the mean and subtract it from x to center the data
    x = x - np.mean(x, axis=0)
    return x


def get_covariance(dataset):
    n = len(dataset)
    # Get the covariance matrix using the covariance equation in HW3
    covariance = (1 / (n - 1)) * (np.dot(np.transpose(dataset), dataset))
    return covariance


def get_eig(S, m):
    # Find the eigenvalues and eigenvectors
    Lambda, U = scipy.linalg.eigh(S, subset_by_index=[4096 - m, 4095])

    # Change the order of the eigenvalues so they are in descending order
    size = 0
    for i in range(len(Lambda)):
        size = size + 1

    matrixLambda = np.zeros(size)
    for i in range(len(Lambda)):
        matrixLambda[i] = Lambda[size - 1 - i]

    IdentityMatrix = np.zeros([size, size])
    for i in range(len(IdentityMatrix)):
        IdentityMatrix[i][i] = 1

    matrixLambda = matrixLambda * IdentityMatrix
    # Change the order of the eigenvectors so they correspond to the correct eigenvalues
    n = (size - 1) / 2
    print(n)
    for i in range(len(U)):
        for j in range(len(U[0])):
            if (j > n):
                continue

            temp = U[i][j]
            U[i][j] = U[i][size - 1 - j]
            U[i][size - 1 - j] = temp

    return matrixLambda, U


def get_eig_prop(S, prop):
    # Calculate the proportion of variance for the eigenvalues
    sum = np.trace(S)
    x = prop * sum

    # Calculate the covariance matrix for eigenvalues greater than this proportion of variance
    Lambda, U = scipy.linalg.eigh(S, subset_by_value=(x, np.inf))

    # Rearrange the eigenvalues to be in descending order
    size = 0
    for i in range(len(Lambda)):
        size = size + 1

    newLambda = np.zeros(size)
    for i in range(len(Lambda)):
        newLambda[i] = Lambda[size - 1 - i]

    IdentityMatrix = np.zeros([size, size])
    for i in range(len(IdentityMatrix)):
        IdentityMatrix[i][i] = 1

    newLambda = newLambda * IdentityMatrix

    # Rearrange the eigenvectors to correspond to the new order of eigenvalues
    n = (size - 1) / 2
    for i in range(len(U)):
        for j in range(len(U[0])):
            if (j > n):
                continue

            temp = U[i][j]
            U[i][j] = U[i][size - 1 - j]
            U[i][size - 1 - j] = temp

    return newLambda, U


def project_image(image, U):
    # Turn the image into a matrix
    matrixImage = np.matrix(image)
    # Find the projection a
    a = np.dot(np.transpose(U), np.transpose(matrixImage))
    # Find the reconstructed image/projection
    projection = np.dot(np.transpose(a), np.transpose(U))
    return projection


def display_image(orig, proj):
    # 1. Reshape the images to be 64x64:
    newImage = np.reshape(orig, (64, 64))
    newProj = np.reshape(proj, (64, 64))
    # Transpose/rotate the images so they are the correct direction
    newImage = np.transpose(newImage)
    newProj = np.transpose(newProj)
    # Create a figure with each object
    fig, (newImage1, newProj1) = plt.subplots(figsize=(9, 3), ncols=2)
    # Title each plot
    newImage1.set_title('Original')
    newProj1.set_title('Projection')
    # Display the images on the correct axes
    pos1 = newImage1.imshow(newImage, aspect='equal')
    pos2 = newProj1.imshow(newProj, aspect='equal')
    # Add a color bar to each plot
    fig.colorbar(pos1, ax=newImage1)
    fig.colorbar(pos2, ax=newProj1)

    return fig, newImage1, newProj1


#x = load_and_center_dataset("Iris_64x64.npy")
#S = get_covariance(x)
#Lambda, U = get_eig(S, 4)
#print(Lambda)
#print(U)
#LambdaProp, UProp = get_eig_prop(S, 0.07)
#print(LambdaProp)
#print(UProp)
#projection = project_image(x[50], U)
#print(projection)
#fig, ax1, ax2 = display_image(x[50], projection)
#plt.show()
