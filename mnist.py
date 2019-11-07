# this script is used to prepare the data(saving the data in .csv format format)

# Download the zip file from  ::::    http://yann.lecun.com/exdb/mnist/

# Download the file names  :::  train-images-idx3-ubyte.gz     |and|       train-labels-idx1-ubyte.gz

# unzip the file in current directory and run this script to save the .csv file in current directory

# Below script is takien from  :::   https://pjreddie.com/projects/mnist-in-csv/

def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
    f.close()
    o.close()
    l.close()
print('Getting csv....')
convert("train-images-idx3-ubyte", "train-labels-idx1-ubyte",
        "mnist_train.csv", 60000)
print('csv written in disk.')