from keras.datasets import mnist
import matplotlib.pyplot as plt 

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

digit = train_images[4]

# Display the fourth digit
plt.imshow(digit, cmap='Purples')
plt.show()
