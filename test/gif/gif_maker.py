import imageio as imageio

with imageio.get_writer('result.gif', mode='I') as writer:
    for filename in range(2, 32):
        image = imageio.imread('result' + str(filename) + '.png')
        writer.append_data(image)
    writer.close()