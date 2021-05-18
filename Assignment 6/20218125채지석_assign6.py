import numpy as np
import matplotlib.pyplot as plt

def reconst(A, percent):
    
    U, S, V = np.linalg.svd(A)
    shape_A = np.shape(A)
    full_data_num = np.shape(S)[0]
    trunc_S = np.copy(S)
    remaining_data_num = full_data_num * percent / 100

    for i in range(full_data_num):
        if i > remaining_data_num:
            trunc_S[i] = 0

    new_S = np.zeros(shape_A)     
    np.fill_diagonal(new_S, trunc_S)

    remaining_percent = np.linalg.norm(trunc_S) / np.linalg.norm(S) * 100

    new_A = U @ new_S @ V

    # Normalizing process : to use the function /imsave/ we need
    # to normalize floats into the range of [0,1].
    minimum = np.min(new_A)
    new_A = new_A - minimum
    maximum = np.max(new_A)
    new_A = new_A * (1/maximum)
    
    return (new_A, remaining_percent)


    
fig, ax = plt.subplots()
image = plt.imread("./image.jpg")

red = image[:,:,0]
green = image[:,:,1]
blue = image[:,:,2]

reconstructed_image = np.zeros(tuple([3]) + np.shape(image))

percent_list = [10, 25, 50]
for i in range(len(percent_list)):
    percent = percent_list[i]
    new_red, red_percent = reconst(red, percent)
    new_green, green_percent = reconst(green, percent)
    new_blue, blue_percent = reconst(blue, percent)
    reconstructed_image[i] = np.stack((new_red, new_green, new_blue), -1)

    print()
    print(str(percent) + " percent reconstruction.")
    print("Frobenius norm captured from each RGB is : ")
    print("R : %.4f %%, G : %.4f %%, B : %.4f %%" 
          %(red_percent, green_percent, blue_percent)) 

for i in range(len(percent_list)):
    percent = percent_list[i]
    plt.imsave("image_" + str(percent) + ".jpg", reconstructed_image[i])
    
        
    
